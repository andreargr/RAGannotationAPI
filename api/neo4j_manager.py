# -*- coding: utf-8 -*-
"""
neo4j_manager.py
================
Neo4j client for ontology similarity search.
Handles model caching, vector index queries, and class metadata retrieval.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Model registry  (single source of truth — import this in ontology_loader too)
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
# Timing helper
# ---------------------------------------------------------------------------

@contextmanager
def _timer() -> Generator[dict, None, None]:
    """Context manager that records elapsed seconds into a shared dict."""
    record: dict = {}
    t0 = time.perf_counter()
    yield record
    record["seconds"] = round(time.perf_counter() - t0, 4)


# ---------------------------------------------------------------------------
# Neo4j manager
# ---------------------------------------------------------------------------

@dataclass
class Neo4jManager:
    """
    Wraps the Neo4j driver and SentenceTransformer models.

    Models are loaded lazily on first use and cached for the lifetime of
    this object — so repeated calls within the same session pay the load
    cost only once.
    """

    uri: str
    user: str
    password: str
    _model_cache: dict[str, SentenceTransformer] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "Neo4jManager":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self._driver.close()

    # ── Model cache ────────────────────────────────────────────────────────

    def _get_model(self, model_key: str) -> SentenceTransformer:
        if model_key not in self._model_cache:
            model_name = MODEL_REGISTRY[model_key].model_name
            logger.info("Loading model '%s' for the first time…", model_key)
            self._model_cache[model_key] = SentenceTransformer(model_name)
        else:
            logger.debug("Reusing cached model '%s'.", model_key)
        return self._model_cache[model_key]

    def _encode(self, model_key: str, texts: str | list[str]) -> np.ndarray:
        """
        Encode one or more texts with the given model.
        Always uses normalize_embeddings=True so scores are comparable
        across all methods.
        """
        model = self._get_model(model_key)
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        result = model.encode(
            texts,
            batch_size=512,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return result[0] if single else result

    # ── Reads ──────────────────────────────────────────────────────────────

    def get_ontology_hash(self, ontology_id: str) -> Optional[str]:
        result = self._driver.execute_query(
            "MATCH (o:Ontology {id: $id}) RETURN o.summary_hash AS hash",
            id=ontology_id,
        )
        record = result.records[0] if result.records else None
        return record["hash"] if record else None

    def get_ontology_summary(self, ontology_id: str) -> list:
        result = self._driver.execute_query(
            "MATCH (o:Ontology {id: $ontology_id}) RETURN o.summary AS summary",
            ontology_id=ontology_id,
        )
        record = result.records[0] if result.records else None
        if not record or not record["summary"]:
            return []
        try:
            return json.loads(record["summary"])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Could not parse summary for ontology '%s'.", ontology_id)
            return []

    def get_all_ontologies(self) -> list[dict]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (o:Ontology)
            RETURN o.id       AS ontologyId,
                   o.filename AS filename,
                   o.summary  AS summary
            ORDER BY o.id
            """
        )
        result = []
        for record in records:
            try:
                classes = json.loads(record["summary"]) if record["summary"] else []
            except (json.JSONDecodeError, TypeError):
                classes = []
            result.append({
                "ontologyId": record["ontologyId"],
                "classCount": len(classes),
            })
        return result

    def get_classes_metadata(self, class_ids: list[str]) -> list[dict]:
        """
        Fetch label/comment/synonym metadata for a list of class IDs.
        Queries against ClassEmbedding (the shared base label) so no full
        graph scan is performed.
        """
        if not class_ids:
            return []
        result = self._driver.execute_query(
            """
            UNWIND $class_ids AS cid
            MATCH (ce:ClassEmbedding {class_id: cid})
            RETURN DISTINCT
                ce.class_id AS id,
                ce.labels   AS labels,
                ce.comment  AS comment,
                ce.synonyms AS synonyms
            LIMIT 500
            """,
            class_ids=class_ids,
        )
        return [dict(r) for r in result.records]

    # ── Similarity search ──────────────────────────────────────────────────

    def find_most_similar_ontology(
        self,
        input_text: str,
        model_key: str,
        top_k: int = 10,
    ) -> dict:
        """
        Embed input_text and query the ontology-level vector index.
        Multi-line input is split, encoded per line, and averaged.
        """
        model_cfg = MODEL_REGISTRY[model_key]
        lines = [line.strip() for line in input_text.splitlines() if line.strip()] or [input_text]

        with _timer() as enc_t:
            embeddings = self._encode(model_key, lines)
            avg_embedding = np.mean(embeddings, axis=0).tolist()

        with _timer() as search_t:
            result = self._driver.execute_query(
                """
                CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                YIELD node AS emb, score
                MATCH (o:Ontology)-[:HAS_EMBEDDING]->(emb)
                RETURN o.id       AS ontologyId,
                       o.filename AS filename,
                       score
                ORDER BY score DESC
                """,
                index_name=model_cfg.index_name,
                top_k=top_k,
                embedding=avg_embedding,
            )
            records = [dict(r) for r in result.records]

        return {
            "results": records,
            "timing": {
                "encoding_seconds": enc_t["seconds"],
                "search_seconds": search_t["seconds"],
                "total_seconds": round(enc_t["seconds"] + search_t["seconds"], 4),
            },
        }

    def find_similar_classes_in_ontologies(
        self,
        query_text: str,
        model_key: str,
        ontology_ids: list[str],
        top_k: int = 3,
        score_threshold: float = 0.5,
    ) -> dict:
        """
        Search class-level embeddings restricted to the given ontology IDs.
        Useful for drilling down after an ontology-level search.
        """
        label = MODEL_REGISTRY[model_key].class_label

        with _timer() as enc_t:
            embedding = self._encode(model_key, query_text).tolist()

        with _timer() as search_t:
            result = self._driver.execute_query(
                f"""
                MATCH (ce:{label})
                WHERE ce.ontology_id IN $ontology_ids
                  AND vector.similarity.cosine(ce.vector, $embedding) >= $score_threshold
                RETURN ce.ontology_id AS ontology_id,
                       ce.class_id    AS class_id,
                       ce.labels      AS labels,
                       ce.comment     AS comment,
                       vector.similarity.cosine(ce.vector, $embedding) AS score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                embedding=embedding,
                ontology_ids=ontology_ids,
                score_threshold=score_threshold,
                top_k=top_k,
            )
            records = [dict(r) for r in result.records]

        return {
            "entity": query_text,
            "matches": records,
            "timing": {
                "encoding_seconds": enc_t["seconds"],
                "search_seconds": search_t["seconds"],
                "total_seconds": round(enc_t["seconds"] + search_t["seconds"], 4),
            },
        }

    def find_ontologies_by_class_similarity(
        self,
        query_text: str,
        model_key: str,
        top_k_ontologies: int = 10,
        top_k_classes: int = 50,
        score_threshold: float = 0.8,
    ) -> dict:
        """
        Find the most relevant ontologies by searching class embeddings across
        all ontologies, then aggregating by best-matching class per ontology.
        More precise than ontology-level search for specific terms.
        """
        label = MODEL_REGISTRY[model_key].class_label

        with _timer() as enc_t:
            embedding = self._encode(model_key, query_text).tolist()

        with _timer() as search_t:
            result = self._driver.execute_query(
                f"""
                MATCH (ce:{label})
                WHERE vector.similarity.cosine(ce.vector, $embedding) >= $score_threshold
                WITH ce,
                     vector.similarity.cosine(ce.vector, $embedding) AS score
                ORDER BY score DESC
                LIMIT $top_k_classes
                RETURN ce.ontology_id AS ontology_id,
                       ce.class_id    AS class_id,
                       ce.labels      AS labels,
                       score
                """,
                embedding=embedding,
                score_threshold=score_threshold,
                top_k_classes=top_k_classes,
            )
            records = [dict(r) for r in result.records]

        # Aggregate — keep only the best-scoring class per ontology
        best_per_ontology: dict[str, dict] = {}
        for r in records:
            oid = r["ontology_id"]
            if oid not in best_per_ontology or r["score"] > best_per_ontology[oid]["score"]:
                best_per_ontology[oid] = {
                    "class_id": r["class_id"],
                    "labels": r.get("labels", []),
                    "score": round(r["score"], 4),
                }

        ranked = sorted(best_per_ontology.items(), key=lambda x: -x[1]["score"])
        results = [
            {"ontologyId": oid, "best_matching_class": data}
            for oid, data in ranked[:top_k_ontologies]
        ]

        return {
            "results": results,
            "timing": {
                "encoding_seconds": enc_t["seconds"],
                "search_seconds": search_t["seconds"],
                "total_seconds": round(enc_t["seconds"] + search_t["seconds"], 4),
            },
        }

    def find_ontologies_covering_classes(
            self,
            class_names: list[str],
            model_key: str,
            score_threshold: float = 0.8,
            top_k_matches_per_class: int = 20,
    ) -> dict:
        """
        For each input class name (e.g. extracted from a PlantUML diagram), find
        every ontology class whose embedding scores >= score_threshold against it.

        Unlike find_ontologies_by_class_similarity (which collapses to a single
        best-matching ontology ranked by score), this method is built to answer
        "which ontologies cover this set of classes, and how many of them".

        All input class names are encoded in a single batched call, then a single
        Cypher query (UNWIND over the input classes) compares each input
        embedding against every ClassEmbedding node and keeps matches above
        score_threshold, capped per input class by top_k_matches_per_class.

        Returns
        -------
        dict with:
          - "matches_by_input_class": dict mapping each input class_name to a list
            of {ontology_id, class_id, labels, score} sorted by score desc.
          - "timing": {encoding_seconds, search_seconds, total_seconds}
        """
        label = MODEL_REGISTRY[model_key].class_label

        if not class_names:
            return {
                "matches_by_input_class": {},
                "timing": {"encoding_seconds": 0.0, "search_seconds": 0.0, "total_seconds": 0.0},
            }

        with _timer() as enc_t:
            embeddings = self._encode(model_key, class_names)
            # Build the UNWIND payload: one row per input class name + its embedding.
            query_rows = [
                {"name": name, "embedding": emb.tolist()}
                for name, emb in zip(class_names, embeddings)
            ]

        with _timer() as search_t:
            result = self._driver.execute_query(
                f"""
                UNWIND $rows AS row
                MATCH (ce:{label})
                WITH row, ce,
                     vector.similarity.cosine(ce.vector, row.embedding) AS score
                WHERE score >= $score_threshold
                WITH row.name AS input_class, ce, score
                ORDER BY input_class, score DESC
                WITH input_class, collect({{
                    ontology_id: ce.ontology_id,
                    class_id: ce.class_id,
                    labels: ce.labels,
                    score: score
                }})[0..$top_k_matches_per_class] AS matches
                RETURN input_class, matches
                """,
                rows=query_rows,
                score_threshold=score_threshold,
                top_k_matches_per_class=top_k_matches_per_class,
            )
            matches_by_input_class: dict[str, list[dict]] = {
                r["input_class"]: [
                    {
                        "ontology_id": m["ontology_id"],
                        "class_id": m["class_id"],
                        "labels": m.get("labels", []),
                        "score": round(m["score"], 4),
                    }
                    for m in r["matches"]
                ]
                for r in result.records
            }
            # Ensure every input class name is present, even with no matches.
            for name in class_names:
                matches_by_input_class.setdefault(name, [])

        return {
            "matches_by_input_class": matches_by_input_class,
            "timing": {
                "encoding_seconds": enc_t["seconds"],
                "search_seconds": search_t["seconds"],
                "total_seconds": round(enc_t["seconds"] + search_t["seconds"], 4),
            },
        }


manager = Neo4jManager(
    uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    user=os.environ.get("NEO4J_USER", "neo4j"),
    password=os.environ.get("NEO4J_PASSWORD", "password123"),
)