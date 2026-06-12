# Ontology Processing and Embedding Storage in Neo4j

This module loads RDF/OWL ontologies, extracts their structure, generates semantic embeddings, and stores them in **Neo4j** using a vector index for similarity-based retrieval.

> **Note:** If you are running the full system via Docker Compose, refer to the [main README](../README.md). The instructions below are intended for manual or development setups only.

---

## Prerequisites

### 1. Neo4j

Neo4j must be installed and running before executing this script.

- Version **5.26.6** or higher (required for vector index support)
- Bolt access at `bolt://localhost:7687` (default)
- Valid credentials

Default connection settings used by the script:

```
URI:      bolt://localhost:7687
User:     neo4j
Password: password123
```

> Adjust these values in the script if your configuration differs.

### 2. Python environment

Python **3.10** is required. Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

Key dependencies: `rdflib`, `sentence-transformers`, `neo4j`.

---

## Directory structure

```
embeddings/
├── ontologies/       # Place ontology files here
│   ├── example.owl
│   ├── example.ttl
│   └── example.rdf
└── get_store_embeddings_configfile.py
└── onfig.json
```

All files placed in `ontologies/` will be processed as independent ontologies. Supported formats: `.ttl`, `.owl`, `.rdf`, `.xml`.

---

## What the script does

1. Loads all ontology files from `ontologies/`
2. Parses each RDF/OWL graph using **rdflib**
3. Extracts:
   - Classes (`owl:Class`) with superclasses and subclasses
   - Object and data properties
   - Individuals
   - Labels, comments, and synonyms
4. Generates a textual summary of each ontology
5. Computes embeddings using `sentence-transformers/all-MiniLM-L6-v2`
6. Creates (or recreates) a **Neo4j vector index** named `ontology-embeddings`
7. Stores one `:Ontology` node per ontology with fields: `id`, `filename`, `content`, `summary`, `embedding`

Processing is parallelised across available CPU cores.

---

## Execution

Once Neo4j is running and ontology files are in place:

```bash
python get_store_embeddings_configfile.py
```

---

## Configuring RDF property mappings (`config.json`)

The script reads `config.json` at startup to determine which RDF properties are used as labels, synonyms, and descriptions for each ontology class. If the file is not found or is malformed, the script falls back to SKOS and RDFS defaults without interrupting execution.

The file contains three lists of RDF property URIs:

```json
{
  "labels": [
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://schema.org/name",
    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P108"
  ],
  "synonyms": [
    "http://www.w3.org/2004/02/skos/core#altLabel",
    "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym",
    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P90",
    "http://purl.obolibrary.org/obo/IAO_0000118"
  ],
  "descriptions": [
    "http://www.w3.org/2004/02/skos/core#definition",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://purl.obolibrary.org/obo/IAO_0000115",
    "http://purl.org/dc/elements/1.1/description",
    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P97"
  ]
}
```

### How priority works

**`labels` and `descriptions` — first match wins.**
The parser traverses the URI list from top to bottom and returns the first literal found for a given class. If a class has both `skos:prefLabel` and `rdfs:label` and `skos:prefLabel` is listed first, that value is always used. If the class has no `skos:prefLabel`, the parser falls through to the next candidate automatically.

**`synonyms` — all matches are collected.**
Unlike labels and descriptions, the parser accumulates all literals found across every URI in the list. A class that defines synonyms via `oboInOwl#hasExactSynonym`, `oboInOwl#hasRelatedSynonym`, and `skos:altLabel` simultaneously will have all of them retrieved. List order does not affect the result beyond implicit deduplication.

### Adapting to your ontologies

To support a property not listed above, add its full URI to the appropriate list. For `labels` and `descriptions`, insert it at the position that reflects the desired priority relative to the existing entries. For `synonyms`, order is irrelevant — add it anywhere in the list.

---

## Notes

- Ontologies that cannot be parsed are skipped with a warning.
- Embeddings are L2-normalised to improve cosine similarity quality.
- The script skips re-processing ontologies whose content has not changed since the last run, based on a SHA-256 hash of the generated summary.
- The script includes commented-out logic to store individual classes as separate Neo4j nodes, which can be enabled if finer-grained retrieval is needed.
