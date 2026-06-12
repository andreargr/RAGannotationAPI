# -*- coding: utf-8 -*-
from rdflib import Graph, RDF, RDFS, OWL, URIRef
from rdflib.namespace import SKOS
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import warnings
import logging
import hashlib
from pathlib import Path

logging.getLogger("rdflib").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="rdflib")

NEO4J_SUMMARY_FOLDER = "./summary"
NEO4J_EMBEDDING_FOLDER = "embeddings"
CONFIG_PATH = "./config.json"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

class ParserConfig:
    """
    Loads user-defined RDF property mappings from config.json.

    Expected structure:
    {
        "labels":       ["<uri>", ...],
        "synonyms":     ["<uri>", ...],
        "descriptions": ["<uri>", ...]
    }

    Priority = list order (first match wins for labels/descriptions).
    Falls back to SKOS + RDFS defaults if the file is missing or malformed.
    """

    _DEFAULTS = {
        "labels": [
            str(SKOS.prefLabel),
            str(RDFS.label),
        ],
        "synonyms": [
            str(SKOS.altLabel),
            str(SKOS.hiddenLabel),
        ],
        "descriptions": [
            str(SKOS.definition),
            str(RDFS.comment),
        ],
    }

    def __init__(self, path: str = CONFIG_PATH):
        raw = {}
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            #print(f"Config loaded from '{path}'")
        except FileNotFoundError:
            print(f"'{path}' not found, using default property mappings.")
        except json.JSONDecodeError as e:
            print(f"Could not parse '{path}': {e}, using default property mappings.")

        self.label_uris: list[URIRef] = [
            URIRef(u) for u in raw.get("labels", self._DEFAULTS["labels"])
        ]
        self.synonym_uris: list[URIRef] = [
            URIRef(u) for u in raw.get("synonyms", self._DEFAULTS["synonyms"])
        ]
        self.description_uris: list[URIRef] = [
            URIRef(u) for u in raw.get("descriptions", self._DEFAULTS["descriptions"])
        ]

    def get_preferred_label(self, g: Graph, uri: URIRef) -> str:
        """First label found, in config priority order."""
        for prop in self.label_uris:
            for obj in g.objects(uri, prop):
                return str(obj)
        try:
            return g.qname(uri)
        except Exception:
            return str(uri)

    def get_all_labels(self, g: Graph, uri: URIRef) -> list[str]:
        labels: set[str] = set()
        for prop in self.label_uris:
            for obj in g.objects(uri, prop):
                labels.add(str(obj))
        return list(labels)

    def get_synonyms(self, g: Graph, uri: URIRef) -> list[str]:
        synonyms: set[str] = set()
        for prop in self.synonym_uris:
            for obj in g.objects(uri, prop):
                synonyms.add(str(obj))
        return list(synonyms)

    def get_description(self, g: Graph, uri: URIRef) -> str:
        """First description found, in config priority order."""
        for prop in self.description_uris:
            for obj in g.objects(uri, prop):
                return str(obj)
        return ""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_class_name(cls: dict) -> str:
    labels = cls.get("label")
    if labels and len(labels) > 0:
        return labels[0]
    return "UnnamedClass"


# ---------------------------------------------------------------------------
# Neo4j manager
# ---------------------------------------------------------------------------

class Neo4jManager:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.auth_data = {
            'uri': uri,
            'database': 'neo4j',
            'user': user,
            'pwd': password,
        }

    def close(self):
        self.driver.close()

    def create_ontology_vector_index(self):
        delete_index = "DROP INDEX `ontology-embeddings` IF EXISTS;"
        check_index_query = """
        SHOW INDEXES YIELD name
        WHERE name = 'ontology-embeddings'
        RETURN name
        """
        create_index_query = """
        CREATE VECTOR INDEX `ontology-embeddings`
        FOR (n:Ontology) ON (n.embedding)
        OPTIONS {indexConfig: {
          `vector.dimensions`: 384,
          `vector.similarity_function`: 'cosine'
        }};
        """
        with self.driver.session() as session:
            session.run(delete_index)
            result = session.run(check_index_query)
            if not result.single():
                session.run(create_index_query)
                print("Vector index 'ontology-embeddings' created successfully.")
            else:
                print("Vector index 'ontology-embeddings' already exists.")

    def store_ontology(self, ontology_id, filename, content, summary, embedding, summary_hash):
        query = """
        MERGE (o:Ontology {id: $id})
        SET o.filename = $filename,
            o.content = $content,
            o.summary = $summary,
            o.summary_hash = $summary_hash,
            o.embedding = $embedding
        """
        self.driver.execute_query(
            query,
            id=ontology_id,
            filename=filename,
            content=content,
            summary=summary,
            summary_hash=summary_hash,
            embedding=embedding,
        )

    def get_ontology_hash(self, ontology_id):
        query = """
        MATCH (o:Ontology {id: $id})
        RETURN o.summary_hash AS hash
        """
        result = self.driver.execute_query(query, id=ontology_id)
        record = result.records[0] if result.records else None
        return record["hash"] if record and record["hash"] else None


# ---------------------------------------------------------------------------
# Graph loading (with owlready2 fallback)
# ---------------------------------------------------------------------------

def get_ontology_list(directory: str = "./ontologies") -> list[dict]:
    dir_path = Path(directory)
    extensions = ["*.ttl", "*.owl", "*.rdf", "*.xml"]
    ontologies = []
    for ext in extensions:
        for file in dir_path.glob(ext):
            ontologies.append({
                "id": file.stem,
                "filename": file.name,
                "path": str(file),
                "content": file.read_text(encoding="utf-8", errors="ignore"),
            })
    return ontologies


def get_graph_owlready(dataset: dict) -> Graph | None:
    """Fallback parser using owlready2 for malformed RDF/XML."""
    try:
        import owlready2
        import tempfile

        onto = owlready2.get_ontology(f"file://{dataset['path']}").load()
        g = Graph()
        with tempfile.NamedTemporaryFile(suffix=".nt", delete=False, mode='w') as tmp:
            tmp_path = tmp.name
        onto.save(file=tmp_path, format="ntriples")
        g.parse(tmp_path, format="nt")
        os.unlink(tmp_path)
        return g
    except Exception as e:
        print(f"?? owlready2 also failed for {dataset['path']}: {e}")
        return None


def get_graph(dataset: dict) -> Graph | None:
    formats_to_try = [None, "xml", "turtle", "n3", "nt"]
    for fmt in formats_to_try:
        try:
            g = Graph()
            if fmt is None:
                g.parse(dataset["path"])
            else:
                g.parse(dataset["path"], format=fmt)
            return g
        except Exception:
            continue

    print(f"?? Trying owlready2 fallback for {dataset['path']}")
    return get_graph_owlready(dataset)


# ---------------------------------------------------------------------------
# RDF extraction (all config-aware)
# ---------------------------------------------------------------------------

def get_domains(g: Graph, prop_uri: URIRef, cfg: ParserConfig) -> list[str]:
    return [
        cfg.get_preferred_label(g, o)
        for o in g.objects(prop_uri, RDFS.domain)
        if isinstance(o, URIRef)
    ]


def get_ranges(g: Graph, prop_uri: URIRef, cfg: ParserConfig) -> list[str]:
    return [
        cfg.get_preferred_label(g, o)
        for o in g.objects(prop_uri, RDFS.range)
        if isinstance(o, URIRef)
    ]


def get_individuals_of_class(g: Graph, class_uri: URIRef, cfg: ParserConfig) -> list[dict]:
    return [
        {"iri": str(ind), "label": cfg.get_preferred_label(g, ind)}
        for ind in g.subjects(RDF.type, class_uri)
        if isinstance(ind, URIRef)
    ]


def _property_list(g: Graph, class_uri: URIRef, prop_type: URIRef, cfg: ParserConfig) -> list[dict]:
    """Collects OWL properties of *prop_type* whose domain is class_uri."""
    result = []
    for p in g.subjects(RDF.type, prop_type):
        if class_uri not in g.objects(p, RDFS.domain):
            continue
        result.append({
            "iri": str(p),
            "label": cfg.get_all_labels(g, p),
            "definition": cfg.get_description(g, p),
            "domain": get_domains(g, p, cfg),
            "range": get_ranges(g, p, cfg),
        })
    return result


def extract_class_structure(g: Graph, class_uri: URIRef, cfg: ParserConfig) -> dict:
    super_classes = [
        cfg.get_preferred_label(g, o)
        for o in g.objects(class_uri, RDFS.subClassOf)
        if isinstance(o, URIRef)
    ]
    sub_classes = [
        cfg.get_preferred_label(g, s)
        for s in g.subjects(RDFS.subClassOf, class_uri)
        if isinstance(s, URIRef)
    ]
    equivalent_classes = [
        cfg.get_preferred_label(g, o)
        for o in g.objects(class_uri, OWL.equivalentClass)
        if isinstance(o, URIRef)
    ]
    disjoint_classes = [
        cfg.get_preferred_label(g, o)
        for o in g.objects(class_uri, OWL.disjointWith)
        if isinstance(o, URIRef)
    ]

    return {
        "id": str(class_uri),
        "type": "Class",
        "label": cfg.get_all_labels(g, class_uri),
        "comment": cfg.get_description(g, class_uri),
        "synonyms": cfg.get_synonyms(g, class_uri),
        "superClasses": super_classes,
        "subClasses": sub_classes,
        "objectProperties": _property_list(g, class_uri, OWL.ObjectProperty, cfg),
        "dataProperties": _property_list(g, class_uri, OWL.DatatypeProperty, cfg),
        "equivalentClasses": equivalent_classes,
        "disjointWith": disjoint_classes,
        "individuals": get_individuals_of_class(g, class_uri, cfg),
    }


def extract_all_classes(g: Graph, cfg: ParserConfig) -> list[dict]:
    return [
        extract_class_structure(g, class_uri, cfg)
        for class_uri in g.subjects(RDF.type, OWL.Class)
        if isinstance(class_uri, URIRef)
    ]


# ---------------------------------------------------------------------------
# Text serialisation for embeddings
# ---------------------------------------------------------------------------

def summary_to_text(classes_summary: list[dict]) -> str:
    texts = []

    for cls in classes_summary:
        labels = cls.get("label", [])
        class_name = ", ".join(labels) if labels else "Unnamed class"
        text = f"Class {class_name}. "

        if cls.get("comment"):
            text += f"Description: {cls['comment']}. "

        if cls.get("superClasses"):
            text += f"Superclasses: {', '.join(cls['superClasses'])}. "

        if cls.get("subClasses"):
            text += f"Subclasses: {', '.join(cls['subClasses'])}. "

        for op in cls.get("objectProperties", []):
            op_label = ", ".join(op.get("label", [])) or "Unnamed object property"
            text += f"Object property {op_label}. "
            if op.get("definition"):
                text += f"Definition: {op['definition']}. "
            if op.get("domain"):
                text += f"Domain: {', '.join(op['domain'])}. "
            if op.get("range"):
                text += f"Range: {', '.join(op['range'])}. "

        for dp in cls.get("dataProperties", []):
            dp_label = ", ".join(dp.get("label", [])) or "Unnamed data property"
            text += f"Data property {dp_label}. "
            if dp.get("definition"):
                text += f"Definition: {dp['definition']}. "
            if dp.get("domain"):
                text += f"Domain: {', '.join(dp['domain'])}. "
            if dp.get("range"):
                text += f"Range: {', '.join(dp['range'])}. "

        if cls.get("individuals"):
            individuals_text = ", ".join(
                f"{ind['label']} ({ind['iri']})" for ind in cls["individuals"]
            )
            text += f"Individuals: {individuals_text}. "

        texts.append(text.strip())

    return "\n".join(texts)


# ---------------------------------------------------------------------------
# Per-ontology processing (runs in a worker process)
# ---------------------------------------------------------------------------

def process_ontology(dataset: dict, neo4j_url: str, neo4j_user: str, neo4j_pwd: str,
                     config_path: str = CONFIG_PATH) -> None:
    """
    `config_path` is forwarded explicitly so each worker process can
    reconstruct ParserConfig independently (no shared memory with main process).
    """
    cfg = ParserConfig(config_path)

    graph = get_graph(dataset)
    if graph is None:
        print(f"Skipping {dataset['id']}: graph could not be parsed.")
        return

    ontology_id = dataset["id"]
    manager = Neo4jManager(neo4j_url, neo4j_user, neo4j_pwd)
    try:
        classes_summary = extract_all_classes(graph, cfg)
        summary_text = summary_to_text(classes_summary)

        summary_path = f"{NEO4J_SUMMARY_FOLDER}/{ontology_id}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(classes_summary, f, indent=2, ensure_ascii=False)

        summary_hash = compute_hash(summary_text)

        # Skip if nothing has changed
        existing_hash = manager.get_ontology_hash(ontology_id)
        if existing_hash == summary_hash:
            print(f"Skipping {ontology_id} (no changes)")
            return

        print(f"Updating {ontology_id} (changed or new)")

        embedding = manager.embedding_model.encode(
            summary_text, normalize_embeddings=True
        ).tolist()

        manager.store_ontology(
            ontology_id=ontology_id,
            filename=dataset["filename"],
            content=dataset["content"],
            summary=json.dumps(classes_summary, ensure_ascii=False),
            embedding=embedding,
            summary_hash=summary_hash,
        )
    finally:
        manager.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    datasets = get_ontology_list()
    print(f"Ontologies found: {len(datasets)}")

    # Validate config once in the main process (gives early feedback)
    ParserConfig(CONFIG_PATH)

    neo4j_uri = os.environ.get('NEO4J_URI', "bolt://localhost:7687")
    neo4j_user = os.environ.get('NEO4J_USER', "neo4j")
    neo4j_password = os.environ.get('NEO4J_PASSWORD', "password123")

    os.makedirs("./summary", exist_ok=True)

    manager = Neo4jManager(neo4j_uri, neo4j_user, neo4j_password)
    manager.create_ontology_vector_index()
    manager.close()

    max_workers = os.cpu_count()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_ontology,
                dataset,
                neo4j_uri,
                neo4j_user,
                neo4j_password,
                CONFIG_PATH,        # forwarded to each worker
            )
            for dataset in datasets
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error: {e}")