import re
from typing import List, Dict, Any, Tuple, Optional
from fastapi import HTTPException,APIRouter,UploadFile, File, Form, status
from typing import Optional, Dict, Any
from neo4j_manager import manager
from ontong_rag.router_ontong_rag import analyze_input_text,parse_free_text_entities,build_semantic_mapping_from_entities
import pandas as pd
from io import StringIO, BytesIO
from collections import Counter
from enum import Enum
from pydantic import conint,confloat

router = APIRouter()

def get_similar_ontologies(
    description_text: Optional[str],
    top_k: int
) -> Dict[str, Any]:

    query_text, analysis = analyze_input_text(description_text)

    top_ontologies = manager.find_most_similar_ontology(query_text, top_k)

    return {
        "results": top_ontologies,
    }

def onto_recommender(df, top_n=0):
    #Initialize data structures
    ontology_usage = Counter()
    result = {}

    for column_name in df.columns:
        recommended_onto= get_similar_ontologies(column_name,1)
        ontology_id = recommended_onto["results"][0]["ontologyId"]
        ontology_usage[ontology_id] += 1
        result[column_name] = {"ontology_id": ontology_id}

    #Rank results
    ontology_usage_rank = sorted(
        [{"ontology_id": k, "count": v} for k, v in ontology_usage.items()],
        key=lambda x: x["count"],
        reverse=True
    )

    # 7️⃣ Ajuste top_n considerando empates
    if top_n == 0:
        ranking = ontology_usage_rank
    else:
        if len(ontology_usage_rank) <= top_n:
            ranking = ontology_usage_rank
        else:
            # Encuentra el conteo mínimo que entra en top_n
            min_count_top_n = ontology_usage_rank[top_n - 1]["count"]
            ranking = [o for o in ontology_usage_rank if o["count"] >= min_count_top_n]

    # 8️⃣ Return results
    return {
        "ontology_usage_rank": ranking,
        "recommendations": result
    }

class LLMModel(str, Enum):
    gpt_4o = "gpt-4o"
    gpt_4_1 = "gpt-4.1"
    gpt_4_1_nano = "gpt-4.1-nano"

@router.post("/columns")
async def column_annotation(
    file: UploadFile = File(..., description="CSV or Excel file with the columns to be annotated."),
    top_class_per_entity: conint(ge=1, le=10) = Form(
        1,
        description=(
                "Maximum number of top ontology candidates (classes or individuals) "
                "to return per input entity (1–10)."
        )
    ),
    ontology_ids: Optional[str] = Form("", description="Comma-separated list of ontology OLS IDs to use for annotation."),
    top_n: Optional[int] = Form(2, description="Number of top ontologies to return. 0 means return all."),
    score_threshold: confloat(ge=0.1, le=1.0) = Form(
        0.5,
        description=(
            "Minimum similarity score (0.1–1.0) required for a class to be included "
            "in the returned context."
        )),
    context: bool = Form(
        False,
        description="If true, include ontology class context in the response."
    ),
    show_all_columns: bool = Form(
            False,
            description="If true, include all rows in the output, even if they have no mapping. If false, only rows with matches are shown."
        ),
):
    """
    This endpoint allows users to annotate the columns of a CSV or Excel file using one or more ontologies.
    Parameters info:
    - `file`: CSV or Excel file with the columns to be annotated. If an Excel workbook has multiple sheets, only the first sheet will be processed by default.
    - `top_n` determines how many top ontologies to return. If top_n > 0, the top N ontologies by usage count are returned, including any ties. If top_n = 0, all ontologies are returned.

    Usage logic:
    - If `ontology_ids` are **not provided**, the function uses the internal recommender to suggest ontologies for annotation.
      In this case, `top_n` should be provided to indicate how many top ontologies to select.
    - If `ontology_ids` **are provided**, these ontologies will be used for annotation directly, and `top_n` will be ignored.
    """
    # 1️⃣ Read CSV file
    try:
        content = await file.read()
        filename = file.filename.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(StringIO(content.decode("utf-8")), sep=";")
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only CSV or Excel files are allowed.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    columns_str = ",".join(df.columns)
    entities = parse_free_text_entities(columns_str)
    if not entities:
        raise HTTPException(
            status_code=400,
            detail=(
                "No entities could be derived from the free-text input. "
                "Provide something like 'Person, Workplace, Skill' or one per line."
            ),
        )

    if ontology_ids:
        ontology_ids = [oid.strip() for oid in ontology_ids.split(",") if oid.strip()]
    else:
        recommender_result = onto_recommender(df,top_n)
        ontology_ids = [item["ontology_id"] for item in recommender_result["ontology_usage_rank"]]

    results_per_ontology = []
    context_items: List[Dict[str, Any]] = []
    seen_class_iris: set = set()

    for oi in ontology_ids:
        ontology_data = manager.get_ontology_summary(oi)

        raw_mapping = build_semantic_mapping_from_entities(
            entities=entities,
            ontology_data=ontology_data,
            topk_per_entity=top_class_per_entity,
            topk_index=10,
        )

        # Filtrado por score manteniendo formato mapping: { entity_name: [matches...] }
        filtered_mapping: Dict[str, List[Dict[str, Any]]] = {}

        for entity_name, matches in (raw_mapping or {}).items():
            kept = []
            for m in matches or []:
                score = float(m.get("score", 0.0))
                if score > score_threshold:
                    item = next(
                        (o for o in ontology_data if o.get("id") == m.get("class_iri")),
                        None
                    )
                    entry = {
                        **{k: v for k, v in m.items() if k not in (
                            "class_iri", "class_label", "individual_iri", "individual_label")},
                        "class": {
                            "iri": m.get("class_iri"),
                            "label": m.get("class_label", []),
                            "comment": item.get("comment") if item else None
                        }
                    }
                    if m.get("type") == "individual":
                        entry["individual"] = {
                            "iri": m.get("individual_iri"),
                            "label": m.get("individual_label")
                        }
                    kept.append(entry)

            # ✅ Si show_all_rows es True, incluimos la fila aunque kept esté vacía
            if kept or show_all_columns:
                filtered_mapping[entity_name] = kept

        # Construcción de context si se desea
        if context:
            for entity_name, matches in filtered_mapping.items():
                for m in matches:
                    class_dict = m.get("class", {})
                    class_iri = class_dict.get("iri")
                    if not class_iri or class_iri in seen_class_iris:
                        continue
                    item = next((o for o in ontology_data if o.get("id") == class_iri), None)
                    if item:
                        seen_class_iris.add(class_iri)
                        context_items.append(item)

        results_per_ontology.append(
            {
                "ontology_id": oi,
                "mapping": filtered_mapping,
                "score_threshold": score_threshold
            }
        )

    response = {
        "ontologies": results_per_ontology
    }

    if context:
        response["context"] = context_items

    return response

@router.post("/rows")
async def row_annotation(
    file: UploadFile = File(..., description="CSV or Excel file with the columns to be annotated."),
    column_name: str = Form(..., description="Name of the column whose rows should be annotated.", examples=["column name"]),
    top_class_per_entity: conint(ge=1, le=10) = Form(
        1,
        description=(
                "Maximum number of top ontology candidates (classes or individuals) "
                "to return per input entity (1–10)."
        )
    ),
    ontology_ids: Optional[str] = Form("", description="Comma-separated list of ontology OLS IDs to use for annotation."),
    top_n: Optional[int] = Form(2, description="Number of top ontologies to return. 0 means return all."),
    score_threshold: confloat(ge=0.1, le=1.0) = Form(
        0.5,
        description=(
            "Minimum similarity score (0.1–1.0) required for a class to be included "
            "in the returned context."
        )),
    context: bool = Form(
        False,
        description="If true, include ontology class context in the response."
    ),
    show_all_rows: bool = Form(
        False,
        description="If true, include all rows in the output, even if they have no mapping. If false, only rows with matches are shown."
    ),

):
    """
    This endpoint allows users to annotate the columns of a CSV or Excel file using one or more ontologies.
    Parameters info:
    - `file`: CSV or Excel file with the columns to be annotated. If an Excel workbook has multiple sheets, only the first sheet will be processed by default.
    - `top_n` determines how many top ontologies to return. If top_n > 0, the top N ontologies by usage count are returned, including any ties. If top_n = 0, all ontologies are returned.

    Usage logic:
    - If `ontology_ids` are **not provided**, the function uses the internal recommender to suggest ontologies for annotation.
      In this case, `top_n` should be provided to indicate how many top ontologies to select.
    - If `ontology_ids` **are provided**, these ontologies will be used for annotation directly, and `top_n` will be ignored.
    """
    # 1️⃣ Read CSV file
    try:
        content = await file.read()
        filename = file.filename.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(StringIO(content.decode("utf-8")), sep=";")
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only CSV or Excel files are allowed.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    # column validation
    # Normalize input
    column_name = column_name.strip()

    # 1️⃣ Empty column name
    if not column_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Parameter 'column_name' must not be empty."
        )

    # 2️⃣ Column does not exist
    if column_name not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": f"Column '{column_name}' not found in dataset.",
                "available_columns": list(df.columns)
            }
        )

    values_str = ",".join(df[column_name].astype(str))
    entities = parse_free_text_entities(values_str)
    if not entities:
        raise HTTPException(
            status_code=400,
            detail=(
                "No entities could be derived from the free-text input. "
                "Provide something like 'Person, Workplace, Skill' or one per line."
            ),
        )

    if ontology_ids:
        ontology_ids = [oid.strip() for oid in ontology_ids.split(",") if oid.strip()]
    else:
        recommender_result = onto_recommender(df,top_n)
        ontology_ids = [item["ontology_id"] for item in recommender_result["ontology_usage_rank"]]

    results_per_ontology = []
    context_items: List[Dict[str, Any]] = []
    seen_class_iris: set = set()

    for oi in ontology_ids:
        ontology_data = manager.get_ontology_summary(oi)

        raw_mapping = build_semantic_mapping_from_entities(
            entities=entities,
            ontology_data=ontology_data,
            topk_per_entity=top_class_per_entity,
            topk_index=10,
        )

        # Filtrado por score manteniendo formato mapping: { entity_name: [matches...] }
        filtered_mapping: Dict[str, List[Dict[str, Any]]] = {}

        for entity_name, matches in (raw_mapping or {}).items():
            kept = []
            for m in matches or []:
                score = float(m.get("score", 0.0))
                if score > score_threshold:
                    item = next(
                        (o for o in ontology_data if o.get("id") == m.get("class_iri")),
                        None
                    )
                    entry = {
                        **{k: v for k, v in m.items() if k not in (
                            "class_iri", "class_label", "individual_iri", "individual_label")},
                        "class": {
                            "iri": m.get("class_iri"),
                            "label": m.get("class_label", []),
                            "comment": item.get("comment") if item else None
                        }
                    }
                    if m.get("type") == "individual":
                        entry["individual"] = {
                            "iri": m.get("individual_iri"),
                            "label": m.get("individual_label")
                        }
                    kept.append(entry)

            # ✅ Si show_all_rows es True, incluimos la fila aunque kept esté vacía
            if kept or show_all_rows:
                filtered_mapping[entity_name] = kept

        # Construcción de context si se desea
        if context:
            for entity_name, matches in filtered_mapping.items():
                for m in matches:
                    class_dict = m.get("class", {})
                    class_iri = class_dict.get("iri")
                    if not class_iri or class_iri in seen_class_iris:
                        continue
                    item = next((o for o in ontology_data if o.get("id") == class_iri), None)
                    if item:
                        seen_class_iris.add(class_iri)
                        context_items.append(item)

        results_per_ontology.append(
            {
                "ontology_id": oi,
                "mapping": filtered_mapping,
                "score_threshold": score_threshold
            }
        )

    response = {
        "ontologies": results_per_ontology
    }

    if context:
        response["context"] = context_items

    return response