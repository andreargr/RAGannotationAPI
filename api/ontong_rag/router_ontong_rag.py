import re
from typing import List, Dict, Any, Tuple, Optional
from neo4j_manager import manager
from fastapi import HTTPException,Query,APIRouter
from sentence_transformers import SentenceTransformer
import numpy as np
from neo4j.exceptions import (
    Neo4jError,
    ServiceUnavailable,
    AuthError,
    ClientError,
    DatabaseError,
    TransientError,
)

router = APIRouter()

ENTITY_RE = re.compile(
    r'(?:entity\s+"([^"]+)"|class\s+(\w+))\s*\{([^}]*)\}',
    re.MULTILINE
)
REL_ENTITY_ATTR_RE = re.compile(
    r'("?[^"]+"?|\w+)\s*::\s*("?[^"]+"?|\w+)\s*[-\.]{2,}\s*("?[^"]+"?|\w+)\s*::\s*("?[^"]+"?|\w+)\s*'
)

REL_CLASS_ASSOC_RE = re.compile(
    r'(\w+)\s*(?:"[^"]*")?\s*[-\.]{2,}\s*(?:"[^"]*")?\s*(\w+)'      # src, dst
    r'(?:\s*:\s*([^\n]+?))?'                                        # label (lazy)
    r'(?=\s+\w+\s*(?:"[^"]*")?\s*[-\.]{2,}|\s*@enduml|\s*$)'        #
)

MULTIPLICITY_TOKEN_RE = re.compile(r'^\s*"\s*[\w\.\*\+]*\s*"\s*$')

def raise_neo4j_http(e: Neo4jError):
    msg = str(e)

    # Service down / DNS / ResolvedIPv4Address / timeouts, etc.
    if isinstance(e, ServiceUnavailable):
        if "ResolvedIPv4Address" in msg:
            raise HTTPException(
                status_code=503,
                detail="Failed to resolve Neo4j server address (ResolvedIPv4Address).",
            )
        raise HTTPException(
            status_code=503,
            detail="Neo4j service is currently unavailable (ServiceUnavailable).",
        )

    # Invalid credentials
    if isinstance(e, AuthError):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials when connecting to Neo4j (AuthError).",
        )

    # User / query errors (bad Cypher, bad parameters, etc.)
    if isinstance(e, ClientError):
        raise HTTPException(
            status_code=400,
            detail=f"Error in Neo4j query (ClientError): {msg}",
        )

    # Internal Neo4j engine errors
    if isinstance(e, DatabaseError):
        raise HTTPException(
            status_code=500,
            detail=f"Internal Neo4j error (DatabaseError): {msg}",
        )

    # Transient errors (locks, temporary issues)
    if isinstance(e, TransientError):
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j is temporarily unavailable (TransientError): {msg}",
        )

    # Any other Neo4jError
    raise HTTPException(
        status_code=500,
        detail=f"Unexpected Neo4j error: {msg}",
    )

def iri_suffix(iri):
    if not iri:
        return ''
    if isinstance(iri, dict):  # soporta objetos del tipo {"iri": "..."}
        iri = iri.get('iri', '')
    if not isinstance(iri, str):
        iri = str(iri)
    part = iri.rsplit('#', 1)[-1]
    part = part.rsplit('/', 1)[-1]
    return camel_to_words(part)


def normalize_multiline(text: str) -> str:
    """
    Convert '\\n' to  '\n'
    """
    if "\\n" in text and "\n" not in text:
        # Caso típico: viene JSON-encoded como una sola línea
        return text.replace("\\n", "\n")
    return text

def parse_plantuml_entities(plantuml_text:str):
    entities = {}
    for m in ENTITY_RE.finditer(plantuml_text):
        name = m.group(1) or m.group(2)
        body = m.group(3)

        iri = None
        typ = None

        # Recorremos las líneas del cuerpo
        for line in body.splitlines():
            line = line.strip()
            if re.match(r'^IRI\s*:', line, re.IGNORECASE):
                iri = line.split(':', 1)[1].strip()
            elif re.match(r'^type\s*:', line, re.IGNORECASE):
                typ = line.split(':', 1)[1].strip()

        entities[name] = {"iri_hint": iri, "type_hint": typ}

    return entities

def parse_free_text_entities(text: str) -> Dict[str, Dict[str, Optional[str]]]:
    # Normalizar separadores a '\n'
    for sep in [",", ";"]:
        text = text.replace(sep, "\n")

    names = [line.strip() for line in text.splitlines() if line.strip()]
    if not names:
        return {}

    entities = {}
    for name in names:
        entities[name] = {
            "iri_hint": None,
            "type_hint": None
        }
    return entities


def _unquote(x: str) -> str:
    x = x.strip()
    if len(x) >= 2 and x[0] == '"' and x[-1] == '"':
        return x[1:-1]
    return x

def parse_plantuml_relations(plantuml_text: str):
    relations = []

    # 1) A::campo -- B::campo
    for m1 in REL_ENTITY_ATTR_RE.finditer(plantuml_text):
        src, src_field, dst, dst_field = map(_unquote, m1.groups())
        relations.append({
            "type": "attr_link",
            "src": src,
            "dst": dst,
            "src_field": src_field,
            "dst_field": dst_field,
            "label": "",
            "raw": plantuml_text[m1.start():m1.end()]
        })

    # 2) A "1" -- "0..*" B : label
    for m2 in REL_CLASS_ASSOC_RE.finditer(plantuml_text):
        a, b, label = m2.groups()

        # Filtrar multiplicidades como "1", "0..*"
        if MULTIPLICITY_TOKEN_RE.match(a):
            continue
        if MULTIPLICITY_TOKEN_RE.match(b):
            continue

        relations.append({
            "type": "class_assoc",
            "src": a,
            "dst": b,
            "src_field": "",
            "dst_field": "",
            "label": (label or "").strip(),
            "raw": plantuml_text[m2.start():m2.end()]
        })
    return relations

def analyze_input_text(description_text: Optional[str]) -> Tuple[str, Dict[str, Any]]:

    if not description_text or not description_text.strip():
        raise HTTPException(
            status_code=400,
            detail="No input provided. The 'description_text' parameter is required and cannot be empty."
        )

    text = description_text.strip()
    text = normalize_multiline(text)

    # Heurística sencilla para ver si "parece" PlantUML
    # Puedes añadir/quitar patrones según tus casos reales
    looks_like_plantuml = any(
        token in text
        for token in (
            "@startuml",
            "@enduml",
            "entity ",
            "class ",
            "::",
            "--",
            "..",
        )
    )

    entities_count = 0
    relations_count = 0
    warnings = []

    if looks_like_plantuml:
        try:
            entities = parse_plantuml_entities(text)
            relations = parse_plantuml_relations(text)
            entities_count = len(entities)
            relations_count = len(relations)

            if entities_count == 0:
                warnings.append(
                    "Input seems to be PlantUML, but no entities were detected."
                )
            if relations_count == 0:
                warnings.append(
                    "Input seems to be PlantUML, but no relations were detected."
                )

        except Exception as e:
            warnings.append(
                f"Input looks like PlantUML, but an error occurred while parsing: {e}"
            )

    analysis = {
        "is_plantuml": looks_like_plantuml,
        "entities_count": entities_count,
        "relations_count": relations_count,
        "warnings": warnings,
    }

    return text, analysis

def parse_free_text_entities(text: str) -> Dict[str, Dict[str, Optional[str]]]:
    # Normalizar separadores a '\n'
    for sep in [",", ";"]:
        text = text.replace(sep, "\n")

    names = [line.strip() for line in text.splitlines() if line.strip()]
    if not names:
        return {}

    entities = {}
    for name in names:
        entities[name] = {
            "iri_hint": None,
            "type_hint": None
        }
    return entities

def camel_to_words(s: str):
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', s)
    s = re.sub(r'[_\-]+', ' ', s)
    return ' '.join(s.lower().split())

def class_text_variants(o: dict):
    """
    Genera variantes de texto para una clase usando:
    - label principal
    - sinónimos
    """
    out = []

    # Label principal
    labels = o.get("label") or []
    for l in labels:
        if l:
            out.append((l, "label"))

    # Sinónimos
    synonyms = o.get("synonyms") or []
    for s in synonyms:
        if s:
            out.append((s, "synonym"))

    # Texto compuesto (opcional, todos concatenados)
    comp = " / ".join(labels + synonyms)
    if comp:
        out.append((comp, "composite"))

    # Deduplicar (por texto y tipo)
    seen = set()
    dedup = []
    for t, tag in out:
        key = (t.lower(), tag)
        if key not in seen:
            seen.add(key)
            dedup.append((t, tag))

    return dedup

# def class_and_individual_text_variants(o: dict):
#     out = []
#
#     # 1️⃣ Clase: labels
#     for l in o.get("label") or []:
#         if l:
#             out.append((l, "class_label", "class"))
#
#     # 2️⃣ Clase: sinónimos
#     for s in o.get("synonyms") or []:
#         if s:
#             out.append((s, "class_synonym", "class"))
#
#     # 3️⃣ Individuos
#     for ind in o.get("individuals") or []:
#         ilabel = ind.get("label")
#         if ilabel:
#             out.append((ilabel, "individual_label", "individual"))
#
#     # 4️⃣ Deduplicar
#     seen = set()
#     dedup = []
#     for t, tag, typ in out:
#         key = (t.lower(), tag, typ)
#         if key not in seen:
#             seen.add(key)
#             dedup.append((t, tag, typ))
#
#     return dedup

def class_and_individual_text_variants(o: dict):
    """
    Genera variantes de texto para una clase incluyendo:
    - labels
    - sinónimos
    - comentarios (descripciones)
    - individuos
    """
    out = []

    # 1️⃣ Labels de la clase
    labels = o.get("label") or []
    if isinstance(labels, str):
        labels = [labels]

    for l in labels:
        if l:
            out.append((l, "class", "label"))

    # 2️⃣ Sinónimos
    synonyms = o.get("synonyms") or []
    if isinstance(synonyms, str):
        synonyms = [synonyms]

    for s in synonyms:
        if s:
            out.append((s, "class", "synonym"))

    # 3️⃣ Comentarios / descripciones
    comments = o.get("comment") or o.get("comments") or []
    if isinstance(comments, str):
        comments = [comments]

    for c in comments:
        c = c.strip()
        if c:
            # ⚠️ los comentarios son largos → mejor marcarlos explícitamente
            out.append((c, "class", "comment"))

    # 4️⃣ Individuos
    individuals = o.get("individuals") or []
    for ind in individuals:
        ilabel = ind.get("label")
        if ilabel:
            out.append((ilabel, "individual", "label"))

    # 5️⃣ Texto compuesto (opcional, SOLO clase)
    comp_parts = labels + synonyms
    if comments:
        comp_parts.append(comments[0])  # solo el primer comment
    comp = " / ".join(comp_parts)

    if comp:
        out.append((comp, "class", "composite"))

    # 6️⃣ Deduplicar
    seen = set()
    dedup = []
    for t, ent_type, tag in out:
        key = (t.lower(), ent_type, tag)
        if key not in seen:
            seen.add(key)
            dedup.append((t, ent_type, tag))

    return dedup

def build_embeddings_from_summary(summary: list, model):
    """
    summary: list of classes
    model: instance of SentenceTransformer
    """
    #print("Summary:",summary)
    texts = []
    meta = []

    for i, cls in enumerate(summary):
        for t, ent_type, tag in class_and_individual_text_variants(cls): #get all the ways in which the class appears (label+synonyms+comments)
            texts.append(t)
            meta.append((i, ent_type, tag, t)) #map each embedding to its term so that once the comparison of emb is made, we can then know what class/individual it was.

    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings, meta

def cosine_topk(query_emb: np.ndarray, index_emb: np.ndarray, k: int = 5):
    """

    :param query_emb: query text embedding
    :param index_emb: summary emb
    :param k: number of results
    :return:
    """
    # normalized embeddings → product = cosine
    sims = np.dot(index_emb, query_emb) #calculate similarity
    topk_idx = np.argpartition(-sims, kth=min(k, len(sims)-1))[:k] #get index for the k results
    topk_idx = topk_idx[np.argsort(-sims[topk_idx])] # Sort from highest to lowest score
    return [(int(i), float(sims[i])) for i in topk_idx]

def aggregate_by_ontology_with_individuals(
    top_hits,
    meta,
    ontology,
    limit: int = 3
):
    """
    :param top_hits: list of index
    :param meta: emb mappings
    :param ontology: summary of the ontolofy
    :param limit: number of results
    :return:
    """
    best = {}

    for row_idx, score in top_hits:
        ont_idx, ent_type, tag, text = meta[row_idx] #Retrieve metadata from the hit

        key = (ont_idx, ent_type) #group by class as the same semantic candidate
        if key not in best or score > best[key]["score"]: #Keep only the best score per entity
            best[key] = {
                "score": score,
                "matched_text": text,
                "matched_field": tag,
                "type": ent_type
            }

    ranked = sorted(best.items(), key=lambda kv: -kv[1]["score"])[:limit]
    out = []

    for (ont_idx, ent_type), info in ranked:
        o = ontology[ont_idx]

        result = {
            "type": ent_type,
            "score": round(float(info["score"]), 4),
            "matched_text": info["matched_text"],
            "matched_field": info["matched_field"]
        }

        if ent_type == "class":
            result.update({
                "class_label": o.get("label", []),
                "class_iri": o.get("id")  # <-- ahora viene de 'id' en el summary
            })

        else:  # individual
            ind = next(
                (
                    i for i in o.get("individuals", [])
                    if iri_suffix(i.get("iri", "")).lower() == info["matched_text"].lower()
                       or (i.get("label") or "").lower() == info["matched_text"].lower()
                ),
                None
            )

            result.update({
                "individual_label": ind.get("label") if ind else info["matched_text"],
                "individual_iri": ind.get("iri") if ind else None,
                "class_label": o.get("label"),
                "class_iri": o.get("id"),
            })

        out.append(result)

    return out


def build_semantic_mapping_from_entities(
    entities: Dict[str, Dict[str, Any]],
    ontology_data: List[Dict[str, Any]],
    topk_per_entity: int = 3,
    topk_index: int = 20
):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    index_emb, meta = build_embeddings_from_summary(ontology_data, model) #get normalized emb from the summary

    mapping = {}

    for cls, meta_ent in entities.items():
        qtext = camel_to_words(cls)

        if not qtext or index_emb.shape[0] == 0:
            mapping[cls] = []
            continue

        qemb = model.encode([qtext], convert_to_numpy=True, normalize_embeddings=True)[0] #normalized emb for the query text
        top_hits = cosine_topk(qemb, index_emb, k=topk_index)  #calculate similarity between emb

        mapping[cls] = aggregate_by_ontology_with_individuals(
            top_hits,
            meta,
            ontology_data,
            limit=topk_per_entity
        )
    return mapping

@router.get("/similar-ontologies")
def ontology_similarity(
        top_k: int = Query(5, ge=1, le=100, description="Maximum number of similar ontologies to retrieve (1–100)."),
        blacklist: Optional[str] = Query(
            None,
            description="Comma-separated list of ontology IDs to exclude. If omitted or empty, no ontology is excluded."
        ),
        description_text: Optional[str] = Query(None, description=(
                "Free-text or PlantUML description of what you are looking for.\n"
                "Examples:\n"
                "- 'temperature unit'\n"
                "- 'person profile ontology'\n"
                "- Or a full PlantUML model (@startuml ... @enduml).\n"
                "\n"
                "The text is embedded and compared against ontology embeddings."
        ),
                                                )
):
    """
    This endpoint computes the most similar ontologies for a given textual or PlantUML
    description. The input text is embedded and compared against ontology embeddings
    stored in Neo4j, returning ontology IDs ranked by semantic similarity.

    The input may be:
    - Plain free text (keywords, terms, short descriptions).
    - A complete PlantUML diagram. In this case, the full diagram text is embedded as
      a single query.

    The endpoint does not require valid PlantUML; PlantUML detection is heuristic and
    is used only to generate warnings (e.g., when no entities or relations are found).

    Parameters info:
    - `top_k`: Maximum number of ontology candidates to return (1–100).
    - `description_text`: Raw description (plain text or PlantUML) used as embedding input.

    Usage logic:
    - The function analyzes the input using `analyze_input_text` to detect whether it
      resembles PlantUML and to produce warnings.
    - The raw input text is embedded and passed to `manager.find_most_similar_ontology`.
    - The endpoint returns the ranked list plus PlantUML detection metadata and warnings.
    """

    query_text, analysis = analyze_input_text(description_text) #determinar si es texto plano o un plantUML en base a sus características

    try:
        top_ontologies = manager.find_most_similar_ontology(query_text, top_k)
    except Neo4jError as e:
        raise_neo4j_http(e)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while querying Neo4j: {e}",
        )

    if not top_ontologies:
        raise HTTPException(
            status_code=400,
            detail="No ontology embeddings were found in Neo4j."
        )

    if blacklist:
        blacklist_ids = [x.strip() for x in blacklist.split(",") if x.strip()]
    else:
        blacklist_ids = []

    top_ontologies_filtered = [
        item for item in top_ontologies
        if item["ontologyId"] not in blacklist_ids
    ]

    return {
        "is_plantuml": analysis["is_plantuml"],
        "entities_count": analysis["entities_count"],
        "relations_count": analysis["relations_count"],
        "warnings": analysis["warnings"],
        "results": top_ontologies_filtered,
    }


@router.get("/similar-entities")
def entities_similar(
    description_text: Optional[str] = Query(
        None,
        description=(
            "Input describing entities (classes or individuals).\n"
            "Supported formats:\n"
            "- PlantUML model with classes/entities (@startuml ... @enduml).\n"
            "- Free-text list of entity names, e.g.:\n"
            "    'Person, Workplace, Skill'\n"
            "    'John, Company, Temperature'\n"
            "    or one per line:\n"
            "    Person\\nWorkplace\\nSkill\n"
            "\n"
            "Each entity name is semantically mapped to ontology classes/individuals."
        ),
    ),
    ontology_ids: str = Query(
        None,
        description="One or more ontology IDs, as a comma-separated string, used as search space.",
    ),
    top_class_per_entity: int = Query(
        1,
        ge=1,
        le=10,
        description=(
            "Maximum number of top ontology candidates (classes or individuals) "
            "to return per input entity (1–10)."
        ),
    ),
    score_threshold: float = Query(
        0.5,
        ge=0.1,
        le=1,
        description=(
            "Minimum similarity score (0.1–1.0) required for a class to be included "
            "in the returned context."
        ),
    ),
    context: bool = Query(
        False,
        description="If true, include ontology class context in the response.",
    ),
):
    """
    This endpoint maps input entities to the most semantically similar ontology classes
    and individuals for one or more ontologies.

    The input can be:
    - PlantUML: entities are extracted from class/entity blocks.
    - Free text: each token (comma/semicolon/newline separated) is treated as an entity name.

    Internally, the endpoint builds an embedding index that includes:
    - Class variants (label, synonyms, comments, composite text).
    - Individual variants (individual labels and individual IRI suffixes).

    Parameters info:
    - `description_text`: PlantUML model text or free-text entity list.
    - `ontology_ids`: Comma-separated list of ontology IDs to search in.
    - `top_class_per_entity`: Max number (1–10) of candidates returned per input entity.
    - `score_threshold`: Minimum similarity score to keep a candidate (0.0–1.0).

    Usage logic:
    - The input is analyzed via `analyze_input_text` (PlantUML detection + warnings).
    - If PlantUML:
        • Entities are extracted using `parse_plantuml_entities`.
    - Else:
        • Entities are extracted from free text using `parse_free_text_entities`.
    - For each ontology ID:
        • The ontology structure is loaded using `process_ontology`.
        • Similarity candidates are computed using `build_semantic_mapping_from_entities`.
        • Candidates below `score_threshold` are filtered out.
        • The endpoint also collects a `context` list:
            - For each kept match, the corresponding ontology class entry is added once.
    - The endpoint returns per-ontology mappings and an aggregated class context.
    """

    text, analysis = analyze_input_text(description_text) #check whether the input is a text or a planUML

    ontology_ids_list = [x.strip() for x in ontology_ids.split(",") if x.strip()]

    if not ontology_ids_list:
        raise HTTPException(
            status_code=400,
            detail="At least one ontology id must be provided in 'ontology_ids'.",
        )

    if analysis["is_plantuml"]:
        if analysis["entities_count"] == 0:
            raise HTTPException(
                status_code=400,
                detail="Input looks like PlantUML, but no entities were detected.",
            )
        entities = parse_plantuml_entities(text)
    else:
        #free text
        entities = parse_free_text_entities(text)
        if not entities:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No entities could be derived from the free-text input. "
                    "Provide something like 'Person, Workplace, Skill' or one per line."
                ),
            )
    results_per_ontology = []
    context_items: List[Dict[str, Any]] = []
    seen_class_iris: set = set()

    for ontology_id in ontology_ids_list:
        ontology_data = manager.get_ontology_summary(ontology_id) #get ontology summary

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
                    # Construimos el entry base
                    entry = {
                        # Mantén todos los campos excepto los antiguos planos
                        **{k: v for k, v in m.items() if k not in (
                            "class_iri", "class_label", "individual_iri", "individual_label")},
                        # Diccionario de la clase
                        "class": {
                            "iri": m.get("class_iri"),
                            "label": m.get("class_label", [])
                        }
                    }

                    # Solo añadimos 'individual' si es realmente un individuo
                    if m.get("type") == "individual":
                        entry["individual"] = {
                            "iri": m.get("individual_iri"),
                            "label": m.get("individual_label")
                        }

                    kept.append(entry)

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
                "ontology_id": ontology_id,
                "mapping": filtered_mapping,
                "score_threshold": score_threshold
            }
        )

    response = {
        "is_plantuml": analysis["is_plantuml"],
        "warnings": analysis.get("warnings", []),
        "ontologies": results_per_ontology,
    }

    if context:
        response["context"] = context_items

    return response