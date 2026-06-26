# -*- coding: utf-8 -*-
"""
router_ontong_rag.py
====================
FastAPI router for ontology similarity search endpoints.
"""

from __future__ import annotations

import re
import time
import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Form
from neo4j.exceptions import (
    AuthError,
    ClientError,
    DatabaseError,
    Neo4jError,
    ServiceUnavailable,
    TransientError,
)

from neo4j_manager import manager, MODEL_REGISTRY

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Compiled regular expressions
# ---------------------------------------------------------------------------

ENTITY_RE = re.compile(
    r'(?:entity\s+"([^"]+)"|class\s+(\w+))\s*\{([^}]*)\}',
    re.MULTILINE,
)

REL_ENTITY_ATTR_RE = re.compile(
    r'("?[^"]+"?|\w+)\s*::\s*("?[^"]+"?|\w+)\s*[-\.]{2,}\s*("?[^"]+"?|\w+)\s*::\s*("?[^"]+"?|\w+)\s*'
)

REL_CLASS_ASSOC_RE = re.compile(
    r'(\w+)\s*(?:"[^"]*")?\s*[-\.]{2,}\s*(?:"[^"]*")?\s*(\w+)'
    r'(?:\s*:\s*([^\n]+?))?'
    r'(?=\s+\w+\s*(?:"[^"]*")?\s*[-\.]{2,}|\s*@enduml|\s*$)'
)

MULTIPLICITY_TOKEN_RE = re.compile(r'^\s*"\s*[\w\.\*\+]*\s*"\s*$')

# (se mantiene por compatibilidad; el parser de atributos usa attr_iter_re abajo)
ATTR_RE = re.compile(r'^[\+\-#]?\s*([A-Za-z_]\w*)\s*:\s*([^\s]+)')

# Datatypes XSD que NO deben entrar como texto matchable ni romper el filtro.
_XSD_TYPES = {
    "string", "integer", "int", "float", "double", "decimal", "boolean",
    "datetime", "date", "time", "anyuri", "hexbinary", "long", "short",
    "gyear", "duration",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def validate_model_key(model_key: str) -> None:
    """Raise HTTP 400 if model_key is not registered."""
    if model_key not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model_key '{model_key}'. "
                f"Valid options: {list(MODEL_REGISTRY.keys())}"
            ),
        )


def camel_to_words(s: str) -> str:
    """Convert CamelCase or snake_case identifiers to lower-case words."""
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', s)
    s = re.sub(r'[_\-]+', ' ', s)
    return ' '.join(s.lower().split())


def iri_suffix(iri: object) -> str:
    """Extract and humanise the local name from an IRI string."""
    if not iri:
        return ''
    if isinstance(iri, dict):
        iri = iri.get('iri', '')
    if not isinstance(iri, str):
        iri = str(iri)
    part = iri.rsplit('#', 1)[-1]
    part = part.rsplit('/', 1)[-1]
    return camel_to_words(part)


def normalize_multiline(text: str) -> str:
    """
    Normaliza saltos de linea para el parsing posterior.
    - Unifica CRLF/CR -> \\n
    - Convierte SIEMPRE secuencias escapadas (\\n, \\t) a reales, tambien en el
      caso MIXTO (texto con algunos saltos reales y otros escapados).
    """
    if not text:
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    return text


def raise_neo4j_http(e: Neo4jError) -> None:
    """Map Neo4j exceptions to appropriate HTTP error responses."""
    msg = str(e)
    if isinstance(e, ServiceUnavailable):
        detail = (
            "Failed to resolve Neo4j server address."
            if "ResolvedIPv4Address" in msg
            else "Neo4j service is currently unavailable."
        )
        raise HTTPException(status_code=503, detail=detail)
    if isinstance(e, AuthError):
        raise HTTPException(status_code=401, detail="Invalid credentials when connecting to Neo4j.")
    if isinstance(e, ClientError):
        raise HTTPException(status_code=400, detail=f"Error in Neo4j query: {msg}")
    if isinstance(e, DatabaseError):
        raise HTTPException(status_code=500, detail=f"Internal Neo4j error: {msg}")
    if isinstance(e, TransientError):
        raise HTTPException(status_code=503, detail=f"Neo4j is temporarily unavailable: {msg}")
    raise HTTPException(status_code=500, detail=f"Unexpected Neo4j error: {msg}")


def _unquote(x: str) -> str:
    x = x.strip()
    if len(x) >= 2 and x[0] == '"' and x[-1] == '"':
        return x[1:-1]
    return x


def _local_name(s: str) -> str:
    """Extrae el nombre local de una IRI completa (#, /) o de un prefijo tipo xsd:string."""
    if not s:
        return ""
    s = s.strip()
    for sep in ("#", "/"):
        if sep in s:
            return s.rsplit(sep, 1)[-1].lower()
    return s.split(":")[-1].strip().lower()


def _is_datatype(r: str) -> bool:
    if not r:
        return True
    rl = r.strip().lower()
    return rl.startswith(("xsd:", "xs:", "rdf:", "rdfs:")) or _local_name(r) in _XSD_TYPES


def _norm_dtype(s: str) -> str:
    if not s:
        return ""
    s = _local_name(s)
    if s in {"int", "integer", "long", "short", "byte", "float", "double", "decimal",
             "nonnegativeinteger", "positiveinteger", "unsignedint", "unsignedlong"}:
        return "number"
    if s in {"datetime", "date", "time", "gyear", "gyearmonth"}:
        return "temporal"
    if s in {"anyuri", "hexbinary", "base64binary", "qname", "string",
             "normalizedstring", "token"}:
        return "string"
    return s


# ---------------------------------------------------------------------------
# PlantUML parsers
# ---------------------------------------------------------------------------

def parse_plantuml_entities(plantuml_text: str) -> dict[str, dict]:
    entities: dict[str, dict] = {}
    for m in ENTITY_RE.finditer(plantuml_text):
        name = m.group(1) or m.group(2)
        body = m.group(3)
        iri = typ = None
        for line in body.splitlines():
            line = line.strip()
            if re.match(r'^IRI\s*:', line, re.IGNORECASE):
                iri = line.split(':', 1)[1].strip()
            elif re.match(r'^type\s*:', line, re.IGNORECASE):
                typ = line.split(':', 1)[1].strip()
        entities[name] = {"iri_hint": iri, "type_hint": typ}
    return entities


def parse_plantuml_relations(plantuml_text: str) -> list[dict]:
    relations: list[dict] = []
    for m1 in REL_ENTITY_ATTR_RE.finditer(plantuml_text):
        src, src_field, dst, dst_field = map(_unquote, m1.groups())
        relations.append({
            "type": "attr_link", "src": src, "dst": dst,
            "src_field": src_field, "dst_field": dst_field,
            "label": "", "raw": plantuml_text[m1.start():m1.end()],
        })
    for m2 in REL_CLASS_ASSOC_RE.finditer(plantuml_text):
        a, b, label = m2.groups()
        if MULTIPLICITY_TOKEN_RE.match(a) or MULTIPLICITY_TOKEN_RE.match(b):
            continue
        relations.append({
            "type": "class_assoc", "src": a, "dst": b,
            "src_field": "", "dst_field": "",
            "label": (label or "").strip(),
            "raw": plantuml_text[m2.start():m2.end()],
        })
    return relations


def parse_plantuml_class_attributes(plantuml_text: str) -> dict[str, list[dict[str, str]]]:
    """
    Extrae atributos de clase/entidad de PlantUML.
    Tolerante a que los saltos de linea se hayan perdido/colapsado a espacios
    (usa finditer sobre el cuerpo, no match linea a linea).
    """
    attrs_by_class: dict[str, list[dict[str, str]]] = {}

    # name : tipo   (tipo = token sin espacios; se permite xsd:string, etc.)
    attr_iter_re = re.compile(r'[\+\-#]?\s*([A-Za-z_]\w*)\s*:\s*([^\s{}<]+)')

    for m in ENTITY_RE.finditer(plantuml_text):
        class_name = m.group(1) or m.group(2)
        body = m.group(3)

        # quita estereotipos <<PK>>, <<FK>>... para que no estorben
        body = re.sub(r'<<[^>]*>>', ' ', body)

        for am in attr_iter_re.finditer(body):
            attr_name, attr_type = am.group(1), am.group(2)
            if attr_name.lower() in ("iri", "type"):
                continue
            attrs_by_class.setdefault(class_name, []).append({
                "name": attr_name, "type": attr_type, "raw": am.group(0).strip(),
            })
    return attrs_by_class


def parse_free_text_entities(text: str) -> dict[str, dict[str, Optional[str]]]:
    for sep in [",", ";"]:
        text = text.replace(sep, "\n")
    names = [line.strip() for line in text.splitlines() if line.strip()]
    return {name: {"iri_hint": None, "type_hint": None} for name in names}


def parse_free_text_relations(text: str) -> list[dict[str, str]]:
    for sep in [",", ";"]:
        text = text.replace(sep, "\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [
        {"type": "free_text", "src": "", "dst": "", "src_field": "",
         "dst_field": "", "label": line, "raw": line}
        for line in lines
    ]


def analyze_input_text(description_text: Optional[str]) -> tuple[str, dict]:
    if not description_text or not description_text.strip():
        raise HTTPException(
            status_code=400,
            detail="No input provided. The 'description_text' parameter is required and cannot be empty.",
        )
    text = normalize_multiline(description_text.strip())
    looks_like_plantuml = any(
        token in text
        for token in ("@startuml", "@enduml", "entity ", "class ", "::", "--", "..")
    )
    entities_count = relations_count = 0
    warnings: list[str] = []
    if looks_like_plantuml:
        try:
            entities_count = len(parse_plantuml_entities(text))
            relations_count = len(parse_plantuml_relations(text))
            if entities_count == 0:
                warnings.append("Input seems to be PlantUML, but no entities were detected.")
            if relations_count == 0:
                warnings.append("Input seems to be PlantUML, but no relations were detected.")
        except Exception as exc:
            warnings.append(f"Input looks like PlantUML, but an error occurred while parsing: {exc}")
    return text, {
        "is_plantuml": looks_like_plantuml,
        "entities_count": entities_count,
        "relations_count": relations_count,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def cosine_topk(query_emb: np.ndarray, index_emb: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    sims = np.dot(index_emb, query_emb)
    topk_idx = np.argpartition(-sims, kth=min(k, len(sims) - 1))[:k]
    topk_idx = topk_idx[np.argsort(-sims[topk_idx])]
    return [(int(i), float(sims[i])) for i in topk_idx]


def _deduplicated(texts_and_tags: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for t, tag in texts_and_tags:
        key = (t.lower(), tag)
        if key not in seen:
            seen.add(key)
            out.append((t, tag))
    return out


def object_property_text_variants(
    domain_label: str, domain_iri: str, prop: dict
) -> list[tuple[str, str]]:
    # NOTA: no se tocan object properties (los fixes pedidos son de data props).
    prop_iri = prop.get("iri", "")
    prop_labels = prop.get("label", [])
    prop_label = prop_labels[0] if prop_labels else iri_suffix(prop_iri)
    ranges = prop.get("range", [])
    range_label = ranges[0] if ranges else ""
    definition = prop.get("definition", "")
    dom_label = domain_label or iri_suffix(domain_iri)

    texts: list[tuple[str, str]] = []
    for t in [prop_label, range_label, dom_label]:
        if t:
            texts.append((t, "token"))
    for c in [
        f"{dom_label} {prop_label} {range_label}",
        f"{prop_label} {range_label}",
        f"{dom_label} {prop_label}",
        prop_label,
    ]:
        c = " ".join(c.split())
        if c:
            texts.append((c, "composite"))
    if definition:
        texts.append((f"{prop_label}. {definition}", "definition"))
    if dom_label and range_label:
        texts.append((f"{prop_label} relates {dom_label} to {range_label}", "semantic"))
    return _deduplicated(texts)


def data_property_text_variants(
    domain_label: str, domain_iri: str, prop: dict
) -> list[tuple[str, str]]:
    prop_iri = prop.get("iri", "")
    prop_labels = prop.get("label", [])
    prop_label = prop_labels[0] if prop_labels else iri_suffix(prop_iri)
    ranges = prop.get("range", [])
    range_label = ranges[0] if ranges else ""
    # los datatypes NO entran como texto semantico
    range_text = "" if _is_datatype(range_label) else range_label
    definition = prop.get("definition", "")
    dom_label = domain_label or iri_suffix(domain_iri)

    texts: list[tuple[str, str]] = []
    for t in [prop_label, range_text, dom_label]:
        if t:
            texts.append((t, "token"))
    for c in [f"{dom_label} {prop_label}", f"{prop_label} {range_text}", prop_label]:
        c = " ".join(c.split())
        if c:
            texts.append((c, "composite"))
    if definition:
        texts.append((f"{prop_label}. {definition}", "definition"))
    if dom_label:
        tail = f" is {range_text}" if range_text else ""
        texts.append((f"{prop_label} of {dom_label}{tail}", "semantic"))
    return _deduplicated(texts)


def build_objectprop_index(ontology: list[dict], model) -> tuple[np.ndarray, list]:
    texts: list[str] = []
    meta: list[tuple] = []
    for i, cls in enumerate(ontology):
        domain_labels = cls.get("label", [])
        domain_label = ", ".join(domain_labels) if domain_labels else ""
        domain_iri = cls.get("id", "")
        for j, prop in enumerate(cls.get("objectProperties", []) or []):
            for t, tag in object_property_text_variants(domain_label, domain_iri, prop):
                texts.append(t)
                meta.append((i, j, domain_label, domain_iri, prop.get("iri", ""),
                             prop.get("label", []), prop.get("range", []), tag, t))
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32), []
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return emb, meta


def build_dataprop_index(ontology: list[dict], model) -> tuple[np.ndarray, list]:
    texts: list[str] = []
    meta: list[tuple] = []
    for i, cls in enumerate(ontology):
        domain_labels = cls.get("label", [])
        domain_label = ", ".join(domain_labels) if domain_labels else ""
        domain_iri = cls.get("id", "")
        for j, dp in enumerate(cls.get("dataProperties", []) or []):
            for t, tag in data_property_text_variants(domain_label, domain_iri, dp):
                texts.append(t)
                meta.append((i, j, domain_label, domain_iri, dp.get("iri", ""),
                             dp.get("label", []), dp.get("range", []), tag, t))
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32), []
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return emb, meta


def query_text_for_relation(rel: dict[str, str]) -> str:
    parts = [camel_to_words(rel.get("src", ""))]
    label = rel.get("label", "")
    if label:
        parts.append(camel_to_words(label))
    if rel.get("src_field"):
        parts.append(camel_to_words(rel["src_field"]))
    parts.append(camel_to_words(rel.get("dst", "")))
    if rel.get("dst_field"):
        parts.append(camel_to_words(rel["dst_field"]))
    return " ".join(p for p in parts if p).strip()


def query_text_for_attribute(class_name: str, attr_name: str, attr_type: str = "") -> str:
    # attr_type se ignora a proposito para el embedding: no aporta semantica y
    # arrastra el match hacia cualquier propiedad del mismo datatype.
    parts = [camel_to_words(class_name), camel_to_words(attr_name)]
    return " ".join(p for p in parts if p)


def aggregate_by_object_property(
    top_hits: list[tuple[int, float]],
    meta: list,
    limit: int = 3,
) -> list[dict]:
    best: dict[tuple, dict] = {}
    for row_idx, score in top_hits:
        ont_idx, prop_idx, dom_label, dom_iri, prop_iri, prop_labels, range_labels, tag, text = meta[row_idx]
        key = (ont_idx, prop_idx)
        if key not in best or score > best[key]["score"]:
            best[key] = {
                "score": score, "domain_label": dom_label, "domain_iri": dom_iri,
                "property_iri": prop_iri, "property_labels": prop_labels or [],
                "range_labels": range_labels or [], "matched_field": tag, "matched_text": text,
            }
    ranked = sorted(best.items(), key=lambda kv: -kv[1]["score"])[:limit]
    return [
        {
            "domain": {"iri": info["domain_iri"], "label": info["domain_label"]},
            "property": {
                "iri": info["property_iri"],
                "label": (
                    ", ".join(info["property_labels"])
                    if info["property_labels"]
                    else iri_suffix(info["property_iri"])
                ),
            },
            "range_label": ", ".join(info["range_labels"]) if info["range_labels"] else "",
            "score": round(float(info["score"]), 4),
            "matched_field": info["matched_field"],
            "matched_text": info["matched_text"],
        }
        for _, info in ranked
    ]


def aggregate_by_data_property(
    top_hits: list[tuple[int, float]],
    meta: list,
    limit: int = 3,
) -> list[dict]:
    best: dict[tuple, dict] = {}
    for row_idx, score in top_hits:
        ont_idx, dp_idx, dom_label, dom_iri, prop_iri, prop_labels, range_labels, tag, text = meta[row_idx]
        key = (ont_idx, dp_idx)
        if key not in best or score > best[key]["score"]:
            best[key] = {
                "score": score, "domain_label": dom_label, "domain_iri": dom_iri,
                "property_iri": prop_iri, "property_labels": prop_labels or [],
                "range_labels": range_labels or [], "matched_field": tag, "matched_text": text,
            }
    ranked = sorted(best.items(), key=lambda kv: -kv[1]["score"])[:limit]
    return [
        {
            "domain_label": info["domain_label"],
            "domain_iri": info["domain_iri"],
            "property": {
                "iri": info["property_iri"],
                "label": (
                    ", ".join(info["property_labels"])
                    if info["property_labels"]
                    else iri_suffix(info["property_iri"])
                ),
            },
            "range_label": ", ".join(info["range_labels"]) if info["range_labels"] else "",
            "score": round(float(info["score"]), 4),
            "matched_field": info["matched_field"],
            "matched_text": info["matched_text"],
        }
        for _, info in ranked
    ]


def rank_individuals_by_similarity(
    entity_text: str,
    individuals: list,
    model,
    top_k: int = 3,
    score_threshold: float = 0.0,
) -> list[dict]:
    labeled = [(ind, ind.get("label", "")) for ind in individuals if ind.get("label")]
    if not labeled:
        return []
    query_emb = model.encode(entity_text, normalize_embeddings=True)
    ind_embs = model.encode(
        [label for _, label in labeled],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    sims = np.dot(ind_embs, query_emb)
    ranked = sorted(zip(sims, [ind for ind, _ in labeled]), key=lambda x: x[0], reverse=True)
    return [
        {"iri": ind.get("iri"), "label": ind.get("label"), "score": round(float(score), 4)}
        for score, ind in ranked
        if score >= score_threshold
    ][:top_k]

# ---------------------------------------------------------------------------
# Endpoints  (POST + Form)
# ---------------------------------------------------------------------------

@router.post("/similar-ontologies")
def ontology_similarity(
    minimum_class_coverage: int = Form(
        1, ge=1,
        description="Minimum number of input PlantUML classes an ontology must cover to be returned.",
    ),
    top_k: Optional[int] = Form(
            None, ge=1, le=100,
            description=(
                "Optional cap on how many ontologies to return, applied AFTER the "
                "coverage filter. Useful when many ontologies tie on coverage; the "
                "best-scoring ones are kept. If omitted, all matching ontologies are returned."
            ),
    ),
    blacklist: Optional[str] = Form(None, description="Comma-separated list of ontology IDs to exclude."),
    description_text: Optional[str] = Form(None, description="PlantUML diagram (@startuml ... @enduml) with at least one class/entity "
        "block."),
    search_mode: str = Form(
            "per_class",
            description="'per_class' (coverage by PlantUML class) or 'ontology' (similarity against each ontology's summary embedding).",
    ),
    model_key: str = Form("minilm", description="Embedding model to use."),
    timing: bool = Form(False, description="If True, includes timing breakdown in the response."),
    class_score_threshold: float = Form(
        0.8, ge=0.0, le=1.0,
        description=(
            "Minimum cosine similarity for a PlantUML class to count as covered by an ontology class."
        ),
    ),
    include_class_matches: bool = Form(
        False,
        description="Include PlantUML class -> ontology class mappings."
    ),
):
    """
Recommends ontologies for the input PlantUML, with two selectable modes (`search_mode`):

- `"per_class"` (default): the diagram is split into classes and each ontology
  is ranked by how many input classes it covers (a class counts as covered
  when some ontology class reaches `class_score_threshold`); ties broken by
  best class score. Items: {ontologyId, covered_classes_count, (class_matches)}.

- `"ontology"`: similarity of the whole input against each ontology's summary
  embedding. Items: {ontologyId, filename, score}.

Only PlantUML input is accepted; free text is rejected with HTTP 400.

Parameters:
- `description_text`: PlantUML diagram (@startuml ... @enduml).
- `search_mode`: "per_class" or "ontology".
- `minimum_class_coverage`: (per_class only) min input classes an ontology must cover.
- `top_k`: cap on results. Optional in per_class (None = all); in ontology mode
    it bounds the vector search (defaults to 10 if omitted).
- `class_score_threshold`: (per_class only) min cosine for a class to be covered.
- `include_class_matches`: (per_class only) include per-class matches.
- `blacklist`: comma-separated ontology IDs to exclude.
- `model_key`: embedding model; must match ingestion.
- `timing`: include timing breakdown.

Response:
- `search_mode_used`: "per_class" or "ontology" (item shape depends on it).
- `results`, `is_plantuml`, `entities_count`, `relations_count`, `warnings`.
    """
    validate_model_key(model_key)
    query_text, analysis = analyze_input_text(description_text)

    if not analysis["is_plantuml"] or analysis["entities_count"] == 0:
        raise HTTPException(
            status_code=400,
            detail=("This endpoint only accepts PlantUML input with at least one "
                    "recognizable class/entity block."),
        )

    blacklist_ids = {x.strip() for x in blacklist.split(",")} if blacklist else set()

    if search_mode not in ("per_class", "ontology"):
        raise HTTPException(status_code=400, detail="search_mode must be 'per_class' or 'ontology'.")

    # ----- MODO ONTOLOGIA  -----
    if search_mode == "ontology":
        try:
            response_data = manager.find_most_similar_ontology(
                input_text=query_text,
                model_key=model_key,
                top_k=top_k if top_k is not None else 10,
            )
        except Neo4jError as e:
            raise_neo4j_http(e)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error while querying Neo4j: {e}")

        results = [o for o in response_data["results"] if o["ontologyId"] not in blacklist_ids]
        result: dict = {
            "search_mode_used": "ontology",
            "is_plantuml": analysis["is_plantuml"],
            "entities_count": analysis["entities_count"],
            "relations_count": analysis["relations_count"],
            "warnings": analysis["warnings"],
            "results": results,
        }
        if timing:
            result["timing"] = response_data["timing"]
        return result

    # ----- MODO PER_CLASS (cobertura) -----
    entities = parse_plantuml_entities(query_text)
    class_names = list(entities.keys())

    try:
        response_data = manager.find_ontologies_covering_classes(
            class_names=class_names,
            model_key=model_key,
            score_threshold=class_score_threshold,
        )
    except Neo4jError as e:
        raise_neo4j_http(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while querying Neo4j: {e}")

    matches_by_input_class = response_data["matches_by_input_class"]

    # Aggregate per ontology: which input classes does it cover, and with what matches.
    per_ontology: dict[str, dict] = {}
    for input_class, matches in matches_by_input_class.items():
        for m in matches:
            oid = m["ontology_id"]
            bucket = per_ontology.setdefault(oid, {"covered": {}, "best_score": 0.0})
            covered_for_class = bucket["covered"].setdefault(input_class, [])
            covered_for_class.append({
                "class_id": m["class_id"],
                "labels": m["labels"],
                "score": m["score"],
            })
            bucket["best_score"] = max(bucket["best_score"], m["score"])
    ranked = sorted(
        per_ontology.items(),
        key=lambda kv: (-len(kv[1]["covered"]), -kv[1]["best_score"]),
    )

    top_ontologies = []
    for oid, data in ranked:
        if len(data["covered"]) < minimum_class_coverage:
            continue
        entry = {
            "ontologyId": oid,
            "covered_classes_count": len(data["covered"]),
        }
        if include_class_matches:
            entry["class_matches"] = [
                {"input_class": input_class, "matches": matches}
                for input_class, matches in data["covered"].items()
            ]
        top_ontologies.append(entry)

    top_ontologies = [o for o in top_ontologies if o["ontologyId"] not in blacklist_ids]

    if top_k is not None:
        top_ontologies = top_ontologies[:top_k]

    result: dict = {
        # "model_key": model_key,
        "search_mode_used": "per_class",
        "is_plantuml": analysis["is_plantuml"],
        "entities_count": analysis["entities_count"],
        "relations_count": analysis["relations_count"],
        "warnings": analysis["warnings"],
        "results": top_ontologies,
    }
    if timing:
        result["timing"] = response_data["timing"]
    return result

@router.post("/similar-entities")
def entities_similar(
    description_text: Optional[str] = Form(None, description="PlantUML or free-text entity list."),
    ontology_ids: Optional[str] = Form(None, description="Comma-separated ontology IDs."),
    top_class_per_entity: int = Form(1, ge=1, le=10),
    score_threshold: float = Form(0.5, ge=0.1, le=1),
    include_individuals: bool = Form(False, description="If True, includes ranked individuals of each matched class."),
    context: bool = Form(False),
    model_key: str = Form("minilm", description="Embedding model to use."),
    timing: bool = Form(False, description="If True, includes timing breakdown in the response."),
):
    """
This endpoint maps each input entity to the most semantically similar ontology
classes within one or more specified ontologies, using pre-computed class embeddings
stored in Neo4j.

The input can be:
- PlantUML: entities are extracted from class/entity blocks (@startuml ... @enduml).
- Free text: each token (comma/semicolon/newline separated) is treated as an entity name.

Each class embedding was computed at ingestion time as the
average vector of its label, synonym, comment, and individual label encodings, and
stored directly in Neo4j. At query time, only the input entity texts are encoded.

Parameters info:
- `description_text`: PlantUML model text or free-text entity list.
- `ontology_ids`: Comma-separated list of ontology IDs to search in. At least one
  is required.
- `top_class_per_entity`: Max number (1–10) of class candidates returned per input
  entity, ordered by descending similarity score.
- `score_threshold`: Minimum cosine similarity score (0.1–1.0) to keep a class match.
  Applied inside Neo4j before results are returned, not in Python.
- `include_individuals`: If True, for each matched class the endpoint fetches its
  individuals from the ontology summary and re-ranks them by cosine similarity to
  the input entity text using the selected model. The top `top_class_per_entity`
  individuals above `score_threshold` are returned under an `individuals` field
  in each class match. Requires one additional Neo4j summary fetch per ontology.
- `context`: If True, the full class metadata (labels, comment, synonyms, individuals,
  object properties, and data properties) is fetched from Neo4j and returned in a
  separate `context` field. If `include_individuals=True` is also set, the summary
  fetch is shared between both features to avoid duplicate queries.
- `model_key`: Embedding model to use ('biolord' or 'minilm'). Must match the model
  used during ingestion for meaningful similarity scores.
- `timing`: If True, includes a timing breakdown per ontology in the response.

Usage logic:
- The input is analyzed via `analyze_input_text` (PlantUML detection + warnings).
- If PlantUML, entities are extracted using `parse_plantuml_entities`.
- If free text, entities are extracted using `parse_free_text_entities`.
- For each ontology ID:
    • If `include_individuals=True` or `context=True`, the ontology summary is
      fetched once from Neo4j and cached for reuse within that ontology.
    • For each entity, `manager.find_similar_classes_in_ontologies` filters
      ClassEmbedding nodes in Neo4j by ontology_id and returns the top matches
      above the score threshold using stored cosine similarity.
    • If `include_individuals=True`, `rank_individuals_by_similarity` re-encodes
      the individuals of each matched class and ranks them by cosine similarity
      to the input entity text.
    • If `context=True`, the full class metadata for each matched class IRI is
      collected from the cached summary and returned in a top-level `context` field.

Timing breakdown (per ontology, accumulated across all entities):
- `encoding_seconds`: total time encoding all entity query texts using the
  sentence transformer.
- `search_seconds`: total time on Neo4j cosine filtering across all entity queries.
- `total_seconds`: sum of the above. Does not include individual re-ranking time
  since that is a lightweight in-memory operation.
"""
    validate_model_key(model_key)

    model = manager._get_model(model_key) if include_individuals else None

    text, analysis = analyze_input_text(description_text)

    ontology_ids_list = [x.strip() for x in (ontology_ids or "").split(",") if x.strip()]
    if not ontology_ids_list:
        raise HTTPException(status_code=400, detail="At least one ontology id must be provided.")

    if analysis["is_plantuml"]:
        if analysis["entities_count"] == 0:
            raise HTTPException(status_code=400, detail="Input looks like PlantUML, but no entities were detected.")
        entities = parse_plantuml_entities(text)
    else:
        entities = parse_free_text_entities(text)
        if not entities:
            raise HTTPException(status_code=400, detail="No entities could be derived from the free-text input.")

    results_per_ontology: list[dict] = []
    timing_per_ontology: list[dict] = []
    context_items: list[dict] = []
    seen_class_iris: set[str] = set()

    for ontology_id in ontology_ids_list:
        total_encoding = 0.0
        total_search = 0.0
        raw_mapping: dict[str, list[dict]] = {}

        for entity_name in entities:
            try:
                response = manager.find_similar_classes_in_ontologies(
                    query_text=entity_name,
                    model_key=model_key,
                    ontology_ids=[ontology_id],
                    top_k=top_class_per_entity,
                    score_threshold=score_threshold,
                )
            except Neo4jError as e:
                raise_neo4j_http(e)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Unexpected error querying classes: {e}")

            total_encoding += response["timing"]["encoding_seconds"]
            total_search += response["timing"]["search_seconds"]
            raw_mapping[entity_name] = response["matches"]

        timing_per_ontology.append({
            "ontology_id": ontology_id,
            "encoding_seconds": round(total_encoding, 4),
            "search_seconds": round(total_search, 4),
            "total_seconds": round(total_encoding + total_search, 4),
        })

        ontology_data_cache: Optional[list] = None
        if include_individuals or context:
            try:
                ontology_data_cache = manager.get_ontology_summary(ontology_id)
            except Neo4jError as e:
                raise_neo4j_http(e)

        filtered_mapping: dict[str, list[dict]] = {}
        for entity_name, matches in raw_mapping.items():
            entries: list[dict] = []
            for m in matches:
                entry: dict = {
                    "score": m["score"],
                    "class": {
                        "iri": m["class_id"],
                        "label": m.get("labels", []),
                        "comment": m.get("comment", ""),
                    },
                }
                if include_individuals and ontology_data_cache is not None and model is not None:
                    class_entry = next(
                        (o for o in ontology_data_cache if o.get("id") == m["class_id"]),
                        None,
                    )
                    raw_individuals = class_entry.get("individuals", []) if class_entry else []
                    entry["individuals"] = rank_individuals_by_similarity(
                        entity_text=entity_name,
                        individuals=raw_individuals,
                        model=model,
                        top_k=top_class_per_entity,
                        score_threshold=score_threshold,
                    )
                entries.append(entry)
            filtered_mapping[entity_name] = entries

        if context and ontology_data_cache is not None:
            for entity_name, matches in filtered_mapping.items():
                for m in matches:
                    class_iri = m.get("class", {}).get("iri")
                    if not class_iri or class_iri in seen_class_iris:
                        continue
                    item = next((o for o in ontology_data_cache if o.get("id") == class_iri), None)
                    if item:
                        seen_class_iris.add(class_iri)
                        context_items.append(item)

        results_per_ontology.append({
            "ontology_id": ontology_id,
            "mapping": filtered_mapping,
            "score_threshold": score_threshold,
        })

    result: dict = {
        # "model_key": model_key,
        "is_plantuml": analysis["is_plantuml"],
        "warnings": analysis.get("warnings", []),
        "ontologies": results_per_ontology,
    }
    if context:
        result["context"] = context_items
    if timing:
        result["timing"] = timing_per_ontology
    return result


@router.post("/similar-relations")
def similar_relation(
    description_text: Optional[str] = Form(None, description="PlantUML or free-text relations."),
    ontology_ids: Optional[str] = Form(None, description="Comma-separated ontology IDs."),
    top_property_per_relation: int = Form(1, ge=1, le=10),
    topk_index: int = Form(30, ge=5, le=200),
    score_threshold: float = Form(0.0, ge=0.0, le=1.0),
    model_key: str = Form("minilm", description="Embedding model to use."),
    timing: bool = Form(False, description="If True, includes timing breakdown in the response."),
):
    """
This endpoint computes semantic similarity between input relationships or attributes
and ontology object/data properties, for one or more specified ontologies.

For each request it fetches the full ontology summary from Neo4j (which includes
objectProperties and dataProperties per class, stored at ingestion time) and builds
in-memory embedding indices on the fly before running cosine search.

The input may be PlantUML (associations/links and class attributes) or free-text
relation descriptions:
- If PlantUML is detected:
    • Object-property relations are extracted from associations/links
      (e.g., ClassA -- ClassB : label).
    • Class attributes are extracted and matched against ontology data properties.
- If free text is provided:
    • Each item is treated as an independent relationship query matched against
      ontology object properties only. No data-property extraction is performed.

Internally, the endpoint builds two in-memory embedding indices per ontology:
1) Object-property index, encoding text variants combining:
   - Domain label / IRI suffix, property label / IRI suffix, range label / IRI suffix.
   - Composite strings: domain+property+range, property+range, domain+property.
   - Semantic sentence: "{property} relates {domain} to {range}".
   - Definition text when available.

2) Data-property index (only when PlantUML with attributes), encoding:
   - Domain label / IRI suffix, property label / IRI suffix, range/datatype label.
   - Composite strings: domain+property, property+range.
   - Semantic sentence: "{property} of {domain} is {range}".
   - Definition text when available.

Parameters info:
- `description_text`: PlantUML model or free-text relations.
- `ontology_ids`: Comma-separated list of ontology IDs to search in.
- `top_property_per_relation`: Max number (1–10) of property matches per relation/attribute.
- `topk_index`: Number of top candidates retrieved from the in-memory index before
  aggregation. Higher values increase recall at the cost of speed.
- `score_threshold`: Minimum cosine similarity score to keep a match. Applied in Python
  after index search.
- `model_key`: Embedding model to use ('biolord' or 'minilm').

Usage logic:
- For each ontology ID:
    • `manager.get_ontology_summary` fetches the full class list including
      objectProperties and dataProperties attached at ingestion time.
    • `build_objectprop_index` encodes all object property text variants into a
      numpy matrix used for cosine search.
    • For each relation, `query_text_for_relation` builds a query string, which is
      encoded and searched via `cosine_topk`, then aggregated by
      `aggregate_by_object_property` and filtered by `score_threshold`.
    • If PlantUML with attributes: `build_dataprop_index` encodes all data property
      text variants. Each attribute is queried similarly via `query_text_for_attribute`,
      `cosine_topk`, and `aggregate_by_data_property`.

Timing breakdown (per ontology):
- `index_build_seconds`: time to encode all object property text variants into the
  in-memory numpy index. This cost is paid once per ontology per request.
- `dp_index_build_seconds`: time to encode all data property text variants into the
  in-memory numpy index. Only non-zero when PlantUML input contains class attributes.
- `query_seconds`: total time to encode all relation/attribute query texts and run
  cosine search against the in-memory indices. Accumulated across all relations
  and all attributes.
- `total_seconds`: sum of all the above.
"""
    validate_model_key(model_key)

    # Reutiliza el modelo cacheado en el manager (no recarga de disco por request)
    model = manager._get_model(model_key)

    text, analysis = analyze_input_text(description_text)

    class_attributes_by_name: dict[str, list] = {}
    if analysis["is_plantuml"]:
        class_attributes_by_name = parse_plantuml_class_attributes(text)

    ontology_ids_list = [x.strip() for x in (ontology_ids or "").split(",") if x.strip()]
    if not ontology_ids_list:
        raise HTTPException(status_code=400, detail="At least one ontology id must be provided.")

    if analysis["is_plantuml"]:
        if analysis["relations_count"] == 0:
            raise HTTPException(status_code=400, detail="Input looks like PlantUML, but no relations were detected.")
        relations = parse_plantuml_relations(text)
    else:
        relations = parse_free_text_relations(text)
        if not relations:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No relations could be derived from the free-text input. "
                    "Provide something like 'Person works at Workplace' or multiple "
                    "relations separated by commas, semicolons or new lines."
                ),
            )

    all_results: list[dict] = []
    timing_per_ontology: list[dict] = []

    for ontology_id in ontology_ids_list:
        try:
            ontology_data = manager.get_ontology_summary(ontology_id)
        except Neo4jError as e:
            raise_neo4j_http(e)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching ontology summary: {e}")

        t_idx = time.perf_counter()
        index_emb, meta = build_objectprop_index(ontology_data, model)
        index_build_time = time.perf_counter() - t_idx

        mapping: list[dict] = []
        query_time = 0.0

        for rel in relations:
            qtext = query_text_for_relation(rel)
            if not qtext or index_emb.shape[0] == 0:
                mapping.append({"raw_relation": rel["raw"], "query_text": qtext, "matches": []})
                continue

            t_q = time.perf_counter()
            try:
                qemb = model.encode([qtext], convert_to_numpy=True, normalize_embeddings=True)[0]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error encoding relation text: {e}")
            top_hits = cosine_topk(qemb, index_emb, k=min(topk_index, max(1, index_emb.shape[0])))
            query_time += time.perf_counter() - t_q

            matches = aggregate_by_object_property(top_hits, meta, limit=top_property_per_relation)
            if score_threshold > 0:
                matches = [m for m in matches if m["score"] >= score_threshold]
            if matches:
                mapping.append({"raw_relation": rel["raw"], "query_text": qtext, "matches": matches})

        properties: list[dict] = []
        dp_index_build_time = 0.0

        if analysis["is_plantuml"] and class_attributes_by_name:
            t_dp = time.perf_counter()
            dp_index_emb, dp_meta = build_dataprop_index(ontology_data, model)
            dp_index_build_time = time.perf_counter() - t_dp

            if dp_index_emb.shape[0] > 0:
                for class_name, attrs in class_attributes_by_name.items():
                    for attr in attrs:
                        attr_name = attr["name"]
                        attr_type = attr["type"]

                        qtext_attr = query_text_for_attribute(class_name, attr_name, attr_type)
                        if not qtext_attr:
                            continue

                        t_q = time.perf_counter()
                        try:
                            qemb_attr = model.encode(
                                [qtext_attr], convert_to_numpy=True, normalize_embeddings=True
                            )[0]
                        except Exception as e:
                            raise HTTPException(status_code=500, detail=f"Error encoding attribute text: {e}")
                        top_hits_dp = cosine_topk(
                            qemb_attr, dp_index_emb,
                            k=min(topk_index, max(1, dp_index_emb.shape[0])),
                        )
                        query_time += time.perf_counter() - t_q

                        # Pedimos mas candidatos para no perder compatibles al filtrar
                        matches_dp = aggregate_by_data_property(
                            top_hits_dp, dp_meta, limit=max(top_property_per_relation * 5, 15)
                        )

                        # Filtro de compatibilidad de tipo (rango desconocido -> no filtra)
                        if attr_type:
                            at = _norm_dtype(attr_type)
                            matches_dp = [
                                m for m in matches_dp
                                if not m.get("range_label")
                                or _norm_dtype(m["range_label"].split(",")[0].strip()) == at
                            ]

                        # Recortamos al limite real DESPUES de filtrar
                        matches_dp = matches_dp[:top_property_per_relation]

                        if score_threshold > 0:
                            matches_dp = [m for m in matches_dp if m["score"] >= score_threshold]

                        if matches_dp:
                            properties.append({
                                "class_name": class_name,
                                "attribute_name": attr_name,
                                "attribute_type": attr_type,
                                "matches": matches_dp,
                            })

        timing_per_ontology.append({
            "ontology_id": ontology_id,
            "index_build_seconds": round(index_build_time, 4),
            "dp_index_build_seconds": round(dp_index_build_time, 4),
            "query_seconds": round(query_time, 4),
            "total_seconds": round(index_build_time + dp_index_build_time + query_time, 4),
        })
        all_results.append({"ontology_id": ontology_id, "relations": mapping, "properties": properties})

    result: dict = {
        # "model_key": model_key,
        "is_plantuml": analysis["is_plantuml"],
        "warnings": analysis.get("warnings", []),
        "ontologies": all_results,
    }
    if timing:
        result["timing"] = timing_per_ontology
    return result