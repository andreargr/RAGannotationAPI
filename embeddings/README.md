# Ontology Processing and Embedding Storage in Neo4j

This module loads RDF/OWL ontologies, extracts their structure, generates semantic
embeddings (at both the **ontology** and the **class** level), and stores everything in
**Neo4j** with the indexes needed for similarity-based retrieval.

> **Note:** If you are running the full system via Docker Compose, refer to the
> [main README](../README.md). The instructions below are intended for **manual /
> development setups only**.

---

## Prerequisites

### 1. Neo4j

Neo4j must be installed and running before executing this script.

- Version **5.26.6** or higher (required for vector index support).
- Bolt access (default `bolt://localhost:7687`).
- Valid credentials.

Connection settings are read from environment variables (with fallbacks):

```
NEO4J_URI       (default: bolt://localhost:7687)
NEO4J_USER      (default: neo4j)
NEO4J_PASSWORD  (default: password)
```

> Set these to match your Neo4j. Prefer exporting the variables (or a `.env`) over
> relying on the in-code defaults, so the same credentials are used everywhere in the
> project. If you connect from your host to a Neo4j running in the Docker stack, use the
> mapped Bolt port `bolt://localhost:8687`; the internal name `neo4j-v2` only resolves
> inside the Docker network.

### 2. Python environment

Python **3.10** is required. Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

Key dependencies: `rdflib`, `sentence-transformers`, `neo4j`, `python-dotenv`, `numpy`.

---

## Directory structure

```
embeddings/
├── ontologies/                      # Place ontology files here
│   ├── example.owl
│   ├── example.ttl
│   └── example.rdf
├── get_store_embeddings_classemb.py # The loader script
└── config.json                      # RDF property mappings (labels/synonyms/descriptions)
```

All files placed in `ontologies/` are processed as independent ontologies. Supported
formats: `.ttl`, `.owl`, `.rdf`, `.xml`.

---

## What the script does

1. Loads all ontology files from `ontologies/`.
2. Parses each RDF/OWL graph using **rdflib** (trying several serialisation formats).
3. For each ontology, extracts:
   - Classes (`owl:Class`) with their labels, comments, and synonyms.
   - Object properties and data properties, attached to a class by their `rdfs:domain`.
   - Individuals (instances) of each class.
4. Builds a textual summary of the ontology.
5. Computes embeddings with `sentence-transformers/all-MiniLM-L6-v2` (dimension 384):
   - **Ontology-level**: one embedding of the summary text.
   - **Class-level**: for each class, the mean of the embeddings of its label(s),
     synonyms, comment, and individual labels.
6. Creates the required Neo4j indexes:
   - A **vector index** `idx-embedding-minilm` on `:OntologyEmbeddingMiniLM(vector)`.
   - A **range index** on `:ClassMiniLM(ontology_id)` (class search is a brute-force
     cosine over the classes of the candidate ontologies, filtered by this index).
7. Stores the data as:
   - `(:Ontology {id, filename, content, summary, summary_hash})`.
   - `(:OntologyEmbedding:OntologyEmbeddingMiniLM {ontology_id, vector, model_key})`,
     linked as `(:Ontology)-[:HAS_EMBEDDING]->(:OntologyEmbedding)`.
   - `(:ClassEmbedding:ClassMiniLM {class_id, ontology_id, vector, labels, comment,
     synonyms})`, linked as `(:Ontology)-[:HAS_CLASS_EMBEDDING]->(:ClassEmbedding)`.

Processing is parallelised across available CPU cores.

---

## Execution

Once Neo4j is running and ontology files are in place:

```bash
python get_store_embeddings_classemb.py
```

By default, ontologies whose content has not changed since the last run are skipped
(based on a SHA-256 hash of the generated summary).

### Resetting

To delete the data created by this loader (`:Ontology`, `:OntologyEmbedding`,
`:ClassEmbedding`) and recompute everything:

```bash
python get_store_embeddings_classemb.py --reset
```

To wipe the **entire** database (destructive; only if the database is dedicated to this):

```bash
python get_store_embeddings_classemb.py --reset-all
```

The same reset can be triggered with the environment variable `RESET_DB=1`.

> When running inside Docker Compose, execute the loader in the API container, where the
> `neo4j-v2` host resolves and `embeddings/` is mounted at `/embeddings`:
>
> ```bash
> docker compose exec api-oskar sh -c "cd /embeddings && python get_store_embeddings_classemb.py"
> ```

---

## Configuring RDF property mappings (`config.json`)

The script reads `config.json` at startup to determine which RDF properties are used as
labels, synonyms, and descriptions for each ontology class. If the file is missing or is
malformed, the script falls back to SKOS and RDFS defaults without interrupting
execution.

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

**`labels` and `descriptions` — first match wins.** The parser traverses the URI list
from top to bottom and returns the first literal found for a given class. If a class has
both `skos:prefLabel` and `rdfs:label` and `skos:prefLabel` is listed first, that value
is used. If the class has no `skos:prefLabel`, the parser falls through to the next
candidate automatically.

**`synonyms` — all matches are collected.** Unlike labels and descriptions, the parser
accumulates every literal found across all URIs in the list. A class that defines
synonyms via `oboInOwl#hasExactSynonym`, `oboInOwl#hasRelatedSynonym`, and
`skos:altLabel` simultaneously will have all of them retrieved. List order does not
affect the result beyond implicit deduplication.

### Adapting to your ontologies

To support a property not listed above, add its full URI to the appropriate list. For
`labels` and `descriptions`, insert it at the position that reflects the desired priority
relative to the existing entries. For `synonyms`, order is irrelevant — add it anywhere.

---

## Notes

- Ontologies that cannot be parsed are skipped with a warning.
- Embeddings are L2-normalised to improve cosine similarity quality.
- Re-processing is skipped for ontologies whose content has not changed since the last
  run (SHA-256 of the generated summary).
