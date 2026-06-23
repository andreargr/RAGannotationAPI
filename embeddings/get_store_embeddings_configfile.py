# -*- coding: utf-8 -*-
"""
ontology_loader.py
==================
Loads OWL/RDF ontologies, extracts class/property metadata, generates
sentence embeddings (per-ontology and per-class), and persists everything
to a Neo4j graph database with vector indexes for similarity search.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
from neo4j import GraphDatabase
from rdflib import Graph, OWL, RDF, RDFS, URIRef
from rdflib.namespace import SKOS
from sentence_transformers import SentenceTransformer
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.getLogger("rdflib").setLevel(logging.ERROR)
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="rdflib")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUMMARY_FOLDER = Path("./summary")
CONFIG_PATH = Path("./config.json")
ONTOLOGIES_DIR = Path("./ontologies")
ONTOLOGY_EXTENSIONS = ("*.ttl", "*.owl", "*.rdf", "*.xml")
PARSE_FORMATS = (None, "xml", "turtle", "n3", "nt")  # None = auto-detect


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    dimensions: int
    ontology_label: str        # label extra del nodo de embedding de ontologia
    index_name: str            # indice VECTORIAL de ontologia
    class_label: str           # label extra del nodo de embedding de clase
    class_index_name: str = ""  # opcional; hoy la clase usa RANGE index, no vectorial


MODEL_REGISTRY: dict[str, ModelConfig] = {
    # "biolord": ModelConfig(
    #     model_name="FremyCompany/BioLORD-2023",
    #     dimensions=768,
    #     ontology_label="OntologyEmbeddingBioLORD",
    #     index_name="idx-embedding-biolord",
    #     class_label="ClassBioLORD",
    #     class_index_name="idx-class-biolord",
    # ),
    "minilm": ModelConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        ontology_label="OntologyEmbeddingMiniLM",
        index_name="idx-embedding-minilm",
        class_label="ClassMiniLM",
        class_index_name="idx-class-minilm",
    ),
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ParserConfig:
    """
    Holds the RDF predicate URIs used to extract labels, synonyms, and
    descriptions from an ontology graph. Falls back to SKOS/RDFS defaults
    when no config file is found or the file is malformed.
    """

    label_uris: list[URIRef] = field(default_factory=lambda: [
        URIRef(str(SKOS.prefLabel)),
        URIRef(str(RDFS.label)),
    ])
    synonym_uris: list[URIRef] = field(default_factory=lambda: [
        URIRef(str(SKOS.altLabel)),
        URIRef(str(SKOS.hiddenLabel)),
    ])
    description_uris: list[URIRef] = field(default_factory=lambda: [
        URIRef(str(SKOS.definition)),
        URIRef(str(RDFS.comment)),
    ])

    @classmethod
    def from_file(cls, path: Path = CONFIG_PATH) -> "ParserConfig":
        """Load from JSON; log a warning and use defaults on any failure."""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning("Config file '%s' not found — using defaults.", path)
            return cls()
        except json.JSONDecodeError as exc:
            logger.warning("Config file '%s' is invalid JSON (%s) — using defaults.", path, exc)
            return cls()

        def _uris(key: str, default: list[URIRef]) -> list[URIRef]:
            return [URIRef(u) for u in raw.get(key, [])] or default

        return cls(
            label_uris=_uris("labels", cls.__dataclass_fields__["label_uris"].default_factory()),
            synonym_uris=_uris("synonyms", cls.__dataclass_fields__["synonym_uris"].default_factory()),
            description_uris=_uris("descriptions", cls.__dataclass_fields__["description_uris"].default_factory()),
        )

    # ── RDF helpers ────────────────────────────────────────────────────────

    def get_preferred_label(self, g: Graph, uri: URIRef) -> str:
        for prop in self.label_uris:
            for obj in g.objects(uri, prop):
                return str(obj)
        try:
            return g.qname(uri)
        except Exception:
            return str(uri)

    def get_all_labels(self, g: Graph, uri: URIRef) -> list[str]:
        return list({str(obj) for prop in self.label_uris for obj in g.objects(uri, prop)})

    def get_synonyms(self, g: Graph, uri: URIRef) -> list[str]:
        return list({str(obj) for prop in self.synonym_uris for obj in g.objects(uri, prop)})

    def get_description(self, g: Graph, uri: URIRef) -> str:
        for prop in self.description_uris:
            for obj in g.objects(uri, prop):
                return str(obj)
        return ""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ClassRecord:
    id: str
    label: list[str]
    comment: str
    synonyms: list[str]
    individuals: list[dict]
    object_properties: list[dict] = field(default_factory=list)
    data_properties: list[dict] = field(default_factory=list)

    def to_text_chunks(self) -> list[str]:
        """Return all text fragments that represent this class for embedding."""
        chunks: list[str] = []
        chunks.extend(lbl for lbl in self.label if lbl)
        chunks.extend(syn for syn in self.synonyms if syn)
        if self.comment:
            chunks.append(self.comment)
        chunks.extend(ind["label"] for ind in self.individuals if ind.get("label"))
        return chunks

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "comment": self.comment,
            "synonyms": self.synonyms,
            "individuals": self.individuals,
            "objectProperties": self.object_properties,
            "dataProperties": self.data_properties,
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def summary_to_text(classes: list[ClassRecord]) -> str:
    """Produce a single string summarising all classes for ontology-level embedding."""
    lines: list[str] = []
    for cls in classes:
        line = f"Class {', '.join(cls.label)}. {cls.comment}"
        for op in cls.object_properties:
            prop_label = ", ".join(op.get("label", [])) or op.get("iri", "")
            line += f" ObjectProperty: {prop_label}."
        for dp in cls.data_properties:
            prop_label = ", ".join(dp.get("label", [])) or dp.get("iri", "")
            line += f" DataProperty: {prop_label}."
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RDF extraction
# ---------------------------------------------------------------------------

def load_graph(path: Path) -> Optional[Graph]:
    """Try multiple RDF serialisation formats; return None on total failure."""
    for fmt in PARSE_FORMATS:
        try:
            g = Graph()
            g.parse(str(path), format=fmt)
            return g
        except Exception as exc:
            logger.debug("Format '%s' failed for '%s': %s", fmt, path.name, exc)
    logger.error("Could not parse '%s' with any known format.", path.name)
    return None


def extract_object_properties(g: Graph, cfg: ParserConfig) -> list[dict]:
    result = []
    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        if not isinstance(prop, URIRef):
            continue
        result.append({
            "iri": str(prop),
            "label": cfg.get_all_labels(g, prop),
            "domain": [str(d) for d in g.objects(prop, RDFS.domain)],
            "range": [str(r) for r in g.objects(prop, RDFS.range)],
            "definition": cfg.get_description(g, prop),
        })
    return result


def extract_data_properties(g: Graph, cfg: ParserConfig) -> list[dict]:
    result = []
    for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
        if not isinstance(prop, URIRef):
            continue
        result.append({
            "iri": str(prop),
            "label": cfg.get_all_labels(g, prop),
            "domain": [str(d) for d in g.objects(prop, RDFS.domain)],
            "range": [str(r) for r in g.objects(prop, RDFS.range)],
            "definition": cfg.get_description(g, prop),
        })
    return result


def extract_classes(g: Graph, cfg: ParserConfig) -> list[ClassRecord]:
    classes: list[ClassRecord] = []
    for cls_uri in g.subjects(RDF.type, OWL.Class):
        if not isinstance(cls_uri, URIRef):
            continue
        individuals = [
            {"iri": str(ind), "label": cfg.get_preferred_label(g, ind)}
            for ind in g.subjects(RDF.type, cls_uri)
            if isinstance(ind, URIRef)
        ]
        classes.append(ClassRecord(
            id=str(cls_uri),
            label=cfg.get_all_labels(g, cls_uri),
            comment=cfg.get_description(g, cls_uri),
            synonyms=cfg.get_synonyms(g, cls_uri),
            individuals=individuals,
        ))
    return classes


def attach_properties(
    classes: list[ClassRecord],
    object_props: list[dict],
    data_props: list[dict],
) -> None:
    """Mutate each ClassRecord in-place, attaching the properties whose domain matches."""
    obj_by_domain: dict[str, list[dict]] = defaultdict(list)
    data_by_domain: dict[str, list[dict]] = defaultdict(list)

    for op in object_props:
        for domain_iri in op["domain"]:
            obj_by_domain[domain_iri].append(op)
    for dp in data_props:
        for domain_iri in dp["domain"]:
            data_by_domain[domain_iri].append(dp)

    for cls in classes:
        cls.object_properties = obj_by_domain.get(cls.id, [])
        cls.data_properties = data_by_domain.get(cls.id, [])


# ---------------------------------------------------------------------------
# Neo4j manager
# ---------------------------------------------------------------------------

class Neo4jManager:
    """Encapsulates all database operations: index management, writes, reads."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "Neo4jManager":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Index management ───────────────────────────────────────────────────

    def create_all_indexes(self) -> None:
        for model_key, cfg in MODEL_REGISTRY.items():
            self._create_vector_index(cfg.index_name, cfg.ontology_label, cfg.dimensions)
            self._create_range_index(cfg.class_label, "ontology_id")

    def _create_vector_index(self, index_name: str, label: str, dimensions: int) -> None:
        with self.driver.session() as session:
            session.run(f"DROP INDEX `{index_name}` IF EXISTS;")
            session.run(f"""
                CREATE VECTOR INDEX `{index_name}`
                FOR (e:{label}) ON (e.vector)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dimensions},
                    `vector.similarity_function`: 'cosine'
                }}}};
            """)
        logger.info("Vector index '%s' ready (:%s).", index_name, label)

    def _create_range_index(self, label: str, property_name: str) -> None:
        index_name = f"idx_{label}_{property_name}_filter"
        with self.driver.session() as session:
            session.run(
                f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{label}) ON (n.{property_name});"
            )
        logger.info("Range index '%s' ready (:%s.%s).", index_name, label, property_name)

    # ── Writes ─────────────────────────────────────────────────────────────

    def upsert_ontology(
        self,
        ontology_id: str,
        filename: str,
        content: str,
        summary: str,
        summary_hash: str,
    ) -> None:
        self.driver.execute_query(
            """
            MERGE (o:Ontology {id: $id})
            SET o.filename     = $filename,
                o.content      = $content,
                o.summary      = $summary,
                o.summary_hash = $summary_hash
            """,
            id=ontology_id,
            filename=filename,
            content=content,
            summary=summary,
            summary_hash=summary_hash,
        )

    def upsert_ontology_embedding(
            self, ontology_id: str, model_key: str, embedding: list[float]
    ) -> None:
        label = MODEL_REGISTRY[model_key].ontology_label
        self.driver.execute_query(
            f"""
            MATCH (o:Ontology {{id: $ontology_id}})
            MERGE (e:OntologyEmbedding:{label} {{ontology_id: $ontology_id}})
            SET e.vector     = $embedding,
                e.model_key  = $model_key,
                e.updated_at = datetime()
            MERGE (o)-[:HAS_EMBEDDING]->(e)
            """,
            ontology_id=ontology_id,
            model_key=model_key,
            embedding=embedding,
        )

    def upsert_class_embeddings_batch(
        self,
        ontology_id: str,
        model_key: str,
        records: list[dict],
    ) -> None:
        """
        Batch-write all class embeddings for one model in a single Cypher
        UNWIND call, instead of one round-trip per class.
        """
        label = MODEL_REGISTRY[model_key].class_label
        self.driver.execute_query(
            f"""
            MATCH (o:Ontology {{id: $ontology_id}})
            UNWIND $records AS rec
            MERGE (ce:ClassEmbedding:{label} {{class_id: rec.class_id, ontology_id: $ontology_id}})
            SET ce.vector     = rec.embedding,
                ce.model_key  = $model_key,
                ce.labels     = rec.labels,
                ce.comment    = rec.comment,
                ce.synonyms   = rec.synonyms,
                ce.updated_at = datetime()
            MERGE (o)-[:HAS_CLASS_EMBEDDING]->(ce)
            """,
            ontology_id=ontology_id,
            model_key=model_key,
            records=records,
        )

    # ── Reads ──────────────────────────────────────────────────────────────

    def get_ontology_hash(self, ontology_id: str) -> Optional[str]:
        result = self.driver.execute_query(
            "MATCH (o:Ontology {id: $id}) RETURN o.summary_hash AS hash",
            id=ontology_id,
        )
        record = result.records[0] if result.records else None
        return record["hash"] if record else None

    # ── Cleanup ────────────────────────────────────────────────────────────

    def wipe_ontology_data(self) -> None:
        """
        Borra SOLO lo que crea este loader: Ontology + OntologyEmbedding +
        ClassEmbedding (y sus relaciones). Recomendado antes de recalcular.
        """
        for label in ("ClassEmbedding", "OntologyEmbedding", "Ontology"):
            self.driver.execute_query(f"MATCH (n:{label}) DETACH DELETE n")
        logger.info("Deleted all :Ontology, :OntologyEmbedding and :ClassEmbedding nodes.")

    def wipe_all(self) -> None:
        """Vacia TODA la base de datos. Destructivo; usar solo si la BD esta dedicada."""
        self.driver.execute_query("MATCH (n) DETACH DELETE n")
        logger.info("Database wiped (all nodes deleted).")


# ---------------------------------------------------------------------------
# Ontology discovery
# ---------------------------------------------------------------------------

def discover_ontologies(directory: Path = ONTOLOGIES_DIR) -> list[dict]:
    ontologies: list[dict] = []
    for ext in ONTOLOGY_EXTENSIONS:
        for file in directory.glob(ext):
            ontologies.append({
                "id": file.stem,
                "filename": file.name,
                "path": file,
                "content": file.read_text(encoding="utf-8", errors="ignore"),
            })
    return ontologies


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_ontology(
    dataset: dict,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_pwd: str,
    config_path: Path = CONFIG_PATH,
) -> None:
    """
    Parse one ontology file, compute embeddings for all models, and persist
    results to Neo4j. Skips processing if the content hash is unchanged.
    """
    ontology_id: str = dataset["id"]
    cfg = ParserConfig.from_file(config_path)

    graph = load_graph(dataset["path"])
    if graph is None:
        logger.error("Skipping '%s': could not parse graph.", ontology_id)
        return

    with Neo4jManager(neo4j_uri, neo4j_user, neo4j_pwd) as manager:
        # ── Extract ────────────────────────────────────────────────────────
        classes = extract_classes(graph, cfg)
        object_props = extract_object_properties(graph, cfg)
        data_props = extract_data_properties(graph, cfg)
        attach_properties(classes, object_props, data_props)

        summary_text = summary_to_text(classes)
        summary_hash = compute_sha256(summary_text)

        # ── Skip if unchanged ──────────────────────────────────────────────
        if manager.get_ontology_hash(ontology_id) == summary_hash:
            logger.info("Skipping '%s': content unchanged.", ontology_id)
            return

        logger.info("Processing '%s'…", ontology_id)

        # ── Persist summary ────────────────────────────────────────────────
        classes_as_dicts = [c.to_dict() for c in classes]
        summary_json = json.dumps(classes_as_dicts, ensure_ascii=False)

        SUMMARY_FOLDER.mkdir(parents=True, exist_ok=True)
        (SUMMARY_FOLDER / f"{ontology_id}.json").write_text(
            json.dumps(classes_as_dicts, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        manager.upsert_ontology(
            ontology_id=ontology_id,
            filename=dataset["filename"],
            content=dataset["content"],
            summary=summary_json,
            summary_hash=summary_hash,
        )

        # ── Embed — one model load per model, batch-encode all classes ─────
        for model_key, model_cfg in MODEL_REGISTRY.items():
            model = SentenceTransformer(model_cfg.model_name)

            # Ontology-level embedding
            ontology_embedding = model.encode(
                summary_text, normalize_embeddings=True
            ).tolist()
            manager.upsert_ontology_embedding(ontology_id, model_key, ontology_embedding)

            # Class-level embeddings — batch all text chunks in one encode call
            class_chunk_sizes: list[int] = []
            all_texts: list[str] = []

            for cls in classes:
                chunks = cls.to_text_chunks()
                class_chunk_sizes.append(len(chunks))
                all_texts.extend(chunks)

            if not all_texts:
                continue

            all_embeddings = model.encode(all_texts, normalize_embeddings=True)

            # Slice back and average per class
            batch_records: list[dict] = []
            cursor = 0
            for cls, chunk_size in zip(classes, class_chunk_sizes):
                if chunk_size == 0:
                    continue
                class_matrix = all_embeddings[cursor: cursor + chunk_size]
                cursor += chunk_size
                avg_embedding = np.mean(class_matrix, axis=0).tolist()
                batch_records.append({
                    "class_id": cls.id,
                    "embedding": avg_embedding,
                    "labels": cls.label,
                    "comment": cls.comment,
                    "synonyms": cls.synonyms,
                })

            manager.upsert_class_embeddings_batch(ontology_id, model_key, batch_records)

        logger.info("Finished '%s'.", ontology_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Flags de limpieza:
    #   --reset      -> borra Ontology + OntologyEmbedding + ClassEmbedding (recomendado)
    #   --reset-all  -> vacia TODA la BD (destructivo)
    # Tambien por entorno: RESET_DB=1
    reset_db = ("--reset" in sys.argv) or os.environ.get("RESET_DB", "").lower() in ("1", "true", "yes")
    reset_all = "--reset-all" in sys.argv

    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_pwd = os.environ.get("NEO4J_PASSWORD", "password123")

    # Reset (si se pidio) + indices, antes de descubrir/cargar, para que el
    # reset sea efectivo aunque no haya ficheros que cargar.
    with Neo4jManager(neo4j_uri, neo4j_user, neo4j_pwd) as manager:
        if reset_all:
            manager.wipe_all()
        elif reset_db:
            manager.wipe_ontology_data()
        manager.create_all_indexes()

    datasets = discover_ontologies()
    logger.info("Ontologies found: %d", len(datasets))

    if not datasets:
        logger.warning("No ontology files found in '%s'.", ONTOLOGIES_DIR)
        return

    max_workers = os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_ontology,
                dataset,
                neo4j_uri,
                neo4j_user,
                neo4j_pwd,
                CONFIG_PATH,
            ): dataset["id"]
            for dataset in datasets
        }
        for future in as_completed(futures):
            ontology_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                logger.error("Failed to process '%s': %s", ontology_id, exc, exc_info=True)

if __name__ == "__main__":
    main()