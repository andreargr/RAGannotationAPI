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
└── get_store_embeddings.py
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

## Notes

- Ontologies that cannot be parsed are skipped with a warning.
- Embeddings are L2-normalised to improve cosine similarity quality.
- The script includes commented-out logic to store individual classes as separate Neo4j nodes, which can be enabled if finer-grained retrieval is needed.
