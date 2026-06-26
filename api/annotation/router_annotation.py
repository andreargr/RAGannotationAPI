# -*- coding: utf-8 -*-
"""
router_annotate.py
==================
FastAPI router for ontology-based annotation of tabular data.
Supports column-header annotation and row-value annotation against
pre-computed class embeddings stored in Neo4j.
"""

from __future__ import annotations

import logging
from collections import Counter
from io import BytesIO, StringIO
from typing import Annotated, Any, Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from neo4j.exceptions import Neo4jError
from pydantic import Field
import random

from neo4j_manager import manager, MODEL_REGISTRY
from ontong_rag.router_ontong_rag import (
    analyze_input_text,
    parse_free_text_entities,
    raise_neo4j_http,
    rank_individuals_by_similarity,
    validate_model_key,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_file_as_dataframe(file: UploadFile) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file into a Pandas DataFrame."""
    try:
        content = file.file.read()
        filename = (file.filename or "").lower()

        if filename.endswith(".csv"):
            return pd.read_csv(StringIO(content.decode("utf-8")), sep=",")
        if filename.endswith((".xls", ".xlsx")):
            return pd.read_excel(BytesIO(content))

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file extension. Please upload a CSV or Excel file.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse file: {exc}",
        )


# ---------------------------------------------------------------------------
# Ontology recommendation
# ---------------------------------------------------------------------------

def get_ontology_recommendations(
    texts: list[str],
    model_key: str,
    top_n: int = 0,
    class_score_threshold: float = 0.8,
) -> list[str]:
    usage_counter: Counter[str] = Counter()

    for text in texts:
        try:
            # Stage 1: class-level
            response_data = manager.find_ontologies_by_class_similarity(
                query_text=text,
                model_key=model_key,
                top_k_ontologies=1,
                score_threshold=class_score_threshold,
            )
            # Stage 2: fallback
            if not response_data["results"]:
                response_data = manager.find_most_similar_ontology(
                    input_text=text, model_key=model_key, top_k=1
                )

        except Neo4jError as e:
            raise_neo4j_http(e)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Error querying ontology index for '{text}': {exc}",
            )

        records = response_data.get("results", [])
        if records:
            # ambos modos devuelven 'ontologyId' en el primer nivel
            usage_counter[records[0]["ontologyId"]] += 1

    sorted_usage = usage_counter.most_common()
    if top_n > 0 and len(sorted_usage) > top_n:
        threshold = sorted_usage[top_n - 1][1]
        return [oid for oid, count in sorted_usage if count >= threshold]
    return [oid for oid, _ in sorted_usage]

# ---------------------------------------------------------------------------
# Semantic annotation core
# ---------------------------------------------------------------------------

def _fetch_summary_cached(
    ontology_id: str,
    cache: dict[str, list],
) -> list:
    """Fetch ontology summary from Neo4j, using cache to avoid repeated queries."""
    if ontology_id not in cache:
        try:
            cache[ontology_id] = manager.get_ontology_summary(ontology_id)
        except Neo4jError as e:
            raise_neo4j_http(e)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching summary for ontology '{ontology_id}': {exc}",
            )
    return cache[ontology_id]


def run_semantic_annotation(
    entities: dict,
    ontology_ids: list[str],
    model_key: str,
    top_class_per_entity: int,
    score_threshold: float,
    include_individuals: bool = False,
    include_context: bool = False,
    show_unmatched: bool = False,
    include_timing: bool = False,
) -> dict[str, Any]:
    """
    Core annotation logic: for each entity, query class-level embeddings in
    Neo4j across the given ontologies and build the response structure.
    """
    total_encoding = 0.0
    total_search = 0.0

    model = manager._get_model(model_key) if include_individuals else None
    summary_cache: dict[str, list] = {}

    # Initialise result buckets per ontology
    grouped: dict[str, dict[str, list]] = {oid: {} for oid in ontology_ids}
    candidates_by_entity: dict[str, list] = {}

    for entity in entities:
        try:
            neo4j_response = manager.find_similar_classes_in_ontologies(
                query_text=entity,
                model_key=model_key,
                ontology_ids=ontology_ids,
                top_k=top_class_per_entity,
                score_threshold=score_threshold,
            )
        except Neo4jError as e:
            raise_neo4j_http(e)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Error querying classes for entity '{entity}': {exc}",
            )

        total_encoding += neo4j_response["timing"].get("encoding_seconds", 0.0)
        total_search += neo4j_response["timing"].get("search_seconds", 0.0)
        matches = neo4j_response["matches"]

        # Build context candidates (only when requested)
        if include_context:
            candidates_by_entity[entity] = [
                {
                    "ontology_id": m["ontology_id"],
                    "iri": m["class_id"],
                    "label": m.get("labels", []),
                    "comment": m.get("comment", ""),
                    "score": round(m["score"], 4),
                }
                for m in matches
            ]

        # Populate grouped mapping
        if not matches and show_unmatched:
            for oid in ontology_ids:
                grouped[oid][entity] = []
        else:
            for m in matches:
                oid = m["ontology_id"]
                if oid not in grouped:
                    continue

                entry: dict[str, Any] = {
                    "score": round(m["score"], 4),
                    "class": {
                        "iri": m["class_id"],
                        "label": m.get("labels", []),
                        "comment": m.get("comment", ""),
                    },
                }

                if include_individuals and model is not None:
                    ontology_data = _fetch_summary_cached(oid, summary_cache)
                    class_entry = next(
                        (o for o in ontology_data if o.get("id") == m["class_id"]),
                        None,
                    )
                    raw_individuals = class_entry.get("individuals", []) if class_entry else []
                    entry["individuals"] = rank_individuals_by_similarity(
                        entity_text=entity,
                        individuals=raw_individuals,
                        model=model,
                        top_k=top_class_per_entity,
                        score_threshold=score_threshold,
                    )

                grouped[oid].setdefault(entity, []).append(entry)

    ontologies_result = [
        {
            "ontology_id": oid,
            "score_threshold": score_threshold,
            "mapping": grouped.get(oid, {}),
        }
        for oid in ontology_ids
    ]

    result: dict[str, Any] = {"ontologies": ontologies_result}

    if include_timing:
        result["timing"] = [{
            "encoding_seconds": round(total_encoding, 4),
            "search_seconds": round(total_search, 4),
            "total_seconds": round(total_encoding + total_search, 4),
        }]

    if include_context:
        result["context"] = {
            "description": f"Top {top_class_per_entity} candidates per entity",
            "entities": candidates_by_entity,
        }

    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/ontologies")
async def list_ontologies():
    """
Returns a list of all ontologies currently stored in Neo4j,
with their ID and class count.

Each ontology was ingested from an OWL/RDF/Turtle file and stored with
its full class summary (including labels, comments, synonyms, individuals,
object properties, and data properties per class).

Response fields per ontology:
- `ontologyId`: unique identifier (derived from the filename stem at ingestion).
- `classCount`: number of OWL classes extracted and stored in the summary.
"""
    try:
        ontologies = manager.get_all_ontologies()
    except Neo4jError as e:
        raise_neo4j_http(e)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error fetching ontologies: {exc}")

    return {"count": len(ontologies), "ontologies": ontologies}


@router.post("/annotate/columns")
async def annotate_columns(
    file: UploadFile = File(...),
    top_class_per_entity: Annotated[int, Field(ge=1, le=10)] = Form(1),
    ontology_ids: str = Form(""),
    top_n: int = Form(2),
    score_threshold: Annotated[float, Field(ge=0.1, le=1.0)] = Form(0.5),
    context: bool = Form(False),
    show_all: bool = Form(False),
    timing: bool = Form(False, description="If True, includes timing breakdown in the response."),
    model_key: str = Form("minilm", description="Embedding model to use: 'biolord' or 'minilm'."),
):
    """
Annotates the column headers of an uploaded CSV or Excel file by mapping each
header to the most semantically similar classes in one or more ontologies.

Each column name is treated as an entity and matched against pre-stored class
embeddings in Neo4j. No in-memory re-encoding of the ontology is performed at
request time — all class vectors were computed and stored at ingestion time.

Parameters info:
- `file`: CSV or Excel file whose column headers will be annotated. If an Excel
  workbook has multiple sheets, only the first sheet is processed.
- `top_class_per_entity`: Max number (1–10) of ontology class candidates returned
  per column header.
- `ontology_ids`: Optional comma-separated list of ontology IDs to annotate against.
  If not provided, ontologies are automatically recommended based on the column names.
- `top_n`: When `ontology_ids` is not provided, controls how many top ontologies the
  recommender selects. If top_n > 0, the top N ontologies by column-match count are
  returned, including ties. If top_n = 0, all matched ontologies are returned.
- `score_threshold`: Minimum cosine similarity score (0.1–1.0) to keep a class match.
  Applied inside Neo4j before results are returned.
- `context`: If True, each match includes the full class metadata (IRI, labels,
  comment, score) per entity in a separate `context` field.
- `show_all`: If True, column headers with no matches above the threshold are still
  included in the mapping with an empty list, making unmatched columns visible.
- `model_key`: Embedding model to use ('biolord' or 'minilm'). Must match the model
  used during ingestion for meaningful similarity scores.

Usage logic:
- If `ontology_ids` are provided, they are used directly.
- If not, `get_ontology_recommendations` embeds each column name and queries the
  Neo4j ontology-level vector index to find the most frequently matched ontologies
  across all columns.
- Column headers are parsed via `parse_free_text_entities`.
- For each entity, `manager.find_similar_classes_in_ontologies` is called, which
  filters ClassEmbedding nodes in Neo4j by ontology_id and returns the top matches
  above the score threshold using stored cosine similarity.

Timing breakdown (accumulated across all entities, returned as a single summary):
- `encoding_seconds`: total time spent encoding all column name query texts using
  the sentence transformer.
- `search_seconds`: total time spent on Neo4j cosine filtering across all column
  name queries.
- `total_seconds`: sum of the above.
"""
    validate_model_key(model_key)
    df = load_file_as_dataframe(file)

    target_ids = (
        [i.strip() for i in ontology_ids.split(",") if i.strip()]
        if ontology_ids
        else get_ontology_recommendations(texts=list(df.columns), model_key=model_key, top_n=top_n)
    )

    entities = parse_free_text_entities(",".join(df.columns))
    if not entities:
        raise HTTPException(status_code=400, detail="No valid entities found in column headers.")

    return {
        "model_key": model_key,
        **run_semantic_annotation(
            entities=entities,
            ontology_ids=target_ids,
            model_key=model_key,
            top_class_per_entity=top_class_per_entity,
            score_threshold=score_threshold,
            include_context=context,
            show_unmatched=show_all,
            include_timing=timing,
        ),
    }


@router.post("/annotate/rows")
async def annotate_rows(
    file: UploadFile = File(...),
    column_name: str = Form(...),
    top_class_per_entity: Annotated[int, Field(ge=1, le=10)] = Form(1),
    ontology_ids: str = Form(""),
    top_n: int = Form(2),
    score_threshold: Annotated[float, Field(ge=0.1, le=1.0)] = Form(0.5),
    include_individuals: bool = Form(False, description="If True, includes ranked individuals of each matched class."),
    context: bool = Form(False),
    show_all: bool = Form(False),
    timing: bool = Form(False, description="If True, includes timing breakdown in the response."),
    model_key: str = Form("minilm", description="Embedding model to use: 'biolord' or 'minilm'."),
):
    """
Annotates the unique values within a specific column of an uploaded CSV or Excel
file by mapping each value to the most semantically similar classes in one or more
ontologies, optionally including ranked individuals of each matched class.

Each unique non-null cell value in the specified column is treated as a search entity
and matched against pre-stored class embeddings in Neo4j. All class vectors were computed and
stored at ingestion time as the average of each class's label, synonym, comment, and
individual label encodings.

Parameters info:
- `file`: CSV or Excel file containing the column to annotate. If an Excel workbook
  has multiple sheets, only the first sheet is processed.
- `column_name`: Name of the column whose unique values will be annotated. Must
  exactly match a column name present in the file.
- `top_class_per_entity`: Max number (1–10) of ontology class candidates returned
  per unique value, ordered by descending similarity score.
- `ontology_ids`: Optional comma-separated list of ontology IDs to annotate against.
  If not provided, ontologies are automatically recommended using the unique values
  of the target column as queries against the ontology-level vector index. This means
  ontology selection is driven by the cell content of the annotated column, not the
  schema.
- `top_n`: When `ontology_ids` is not provided, controls how many top ontologies the
  recommender selects. Each unique value from the target column is embedded and matched
  against all ontology embeddings; the ontologies most frequently matched across values
  are selected. To avoid performance issues with high-cardinality columns, only a sample
  of up to 50 unique values is used. If top_n > 0, the top N by match count are
  returned, including ties. If top_n = 0, all matched ontologies are returned.
- `score_threshold`: Minimum cosine similarity score (0.1–1.0) to keep a class match.
  Applied inside Neo4j before results are returned, not in Python. Also used as the
  minimum score threshold when ranking individuals if `include_individuals=True`.
- `include_individuals`: If True, for each matched class the endpoint fetches its
  individuals from the ontology summary and re-ranks them by cosine similarity to
  the input row value using the selected model. The top `top_class_per_entity`
  individuals above `score_threshold` are returned under an `individuals` field
  in each class match. Requires one additional Neo4j summary fetch per ontology,
  cached and reused across all entities within that ontology.
- `context`: If True, a separate `context` field is included in the response
  containing the full class metadata (IRI, labels, comment, score) for each matched
  class, grouped by entity. If `include_individuals=True` is also set, the ontology
  summary fetch is shared between both features to avoid duplicate queries.
- `show_all`: If True, unique values with no matches above the threshold are still
  included in the mapping with an empty list, making unmatched values visible in
  the response.
- `timing`: If True, includes a timing summary in the response covering encoding
  and Neo4j search time accumulated across all row values.
- `model_key`: Embedding model to use ('biolord' or 'minilm'). Must match the model
  used during ingestion for meaningful similarity scores. 'biolord' is recommended
  for biomedical or life-science cell values; 'minilm' is a faster general-purpose
  alternative.

Usage logic:
- The file is parsed into a DataFrame via `load_file_as_dataframe`.
- If `ontology_ids` are provided, they are used directly as annotation targets.
- If not, a sample of up to 50 unique non-null values from the target column is
  extracted and passed to `get_ontology_recommendations`, which queries the Neo4j
  ontology-level vector index once per value and selects the most frequently matched
  ontologies. Ontology selection is therefore driven by the cell content of the
  annotated column.
- For each entity, `manager.find_similar_classes_in_ontologies` filters
  ClassEmbedding nodes in Neo4j by ontology_id and returns the top matches above
  the score threshold using stored cosine similarity.
- If `include_individuals=True`, the ontology summary is fetched once per ontology
  and cached. For each matched class, `rank_individuals_by_similarity` re-encodes
  the class individuals and ranks them by cosine similarity to the row value.
- Results are grouped by ontology and returned as a mapping from each unique row
  value to its list of matched classes, each with IRI, labels, comment, score,
  and optionally ranked individuals.

Timing breakdown (accumulated across all unique row values, returned as a single summary):
- `encoding_seconds`: total time encoding all row value query texts using the
  sentence transformer.
- `search_seconds`: total time on Neo4j cosine filtering across all row value queries.
- `total_seconds`: sum of the above. Does not include individual re-ranking time
  since that is a lightweight in-memory operation.
"""
    validate_model_key(model_key)
    df = load_file_as_dataframe(file)
    clean_col = column_name.strip()

    if not clean_col:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Parameter 'column_name' must not be empty.",
        )
    if clean_col not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": f"Column '{clean_col}' not found.", "available": list(df.columns)},
        )

    unique_values = df[clean_col].dropna().unique().astype(str).tolist()

    sample_size = 50
    sampled = random.sample(unique_values, min(sample_size, len(unique_values)))

    target_ids = (
        [i.strip() for i in ontology_ids.split(",") if i.strip()]
        if ontology_ids
        else get_ontology_recommendations(texts=sampled, model_key=model_key, top_n=top_n)
    )

    entities = parse_free_text_entities(",".join(unique_values))
    if not entities:
        raise HTTPException(status_code=400, detail="No valid entities found in the specified column.")

    return {
        "model_key": model_key,
        **run_semantic_annotation(
            entities=entities,
            ontology_ids=target_ids,
            model_key=model_key,
            top_class_per_entity=top_class_per_entity,
            score_threshold=score_threshold,
            include_context=context,
            show_unmatched=show_all,
            include_timing=timing,
            include_individuals=include_individuals,
        ),
    }