# RAGannotationAPI — API Server

This directory contains the FastAPI application that exposes the RAG annotation endpoints.

> **Note:** If you are running the full system via Docker Compose, refer to the
> [main README](../README.md). The instructions below are intended for **manual /
> development setups only** (running the API directly on your machine, with your own
> Neo4j).

---

## Prerequisites

Before running the API manually, the following must be completed:

1. Neo4j installed and running (version **≥ 5.26.6**, required for vector index support).
2. Ontology files placed in `../embeddings/ontologies/`.
3. Embeddings generated and stored in Neo4j by running the loader:

```bash
python ../embeddings/get_store_embeddings_classemb.py
```

> The API depends on these embeddings to perform semantic search. Without them, the
> annotation endpoints return empty results. See `../embeddings/README.md` for details
> on the loader and on `config.json`.

---

## Installation

From the project root:

```bash
pip install -r requirements.txt
```

Python **3.10** is recommended. Key dependencies: `fastapi`, `uvicorn`, `neo4j`,
`sentence-transformers`, `python-multipart` (required for the form-based endpoints).

---

## Running the API

```bash
cd api
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000` and the interactive documentation
(Swagger UI) at `http://127.0.0.1:8000/docs`.

The `--reload` flag restarts the server automatically on code changes; useful during
development.

> When running through Docker Compose instead, the API is published on host port
> **9000** (`http://localhost:9000/docs`), not 8000. See the main README.

---

## Environment variables

These configure the Neo4j connection. When running with Docker Compose they are taken
automatically from the `.env` file in the project root. For manual execution, export them
in your shell (or load your own `.env`) before starting the server:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
```

| Variable         | Default                  | Description               |
|------------------|--------------------------|---------------------------|
| `NEO4J_URI`      | `bolt://localhost:7687`  | Neo4j Bolt connection URI |
| `NEO4J_USER`     | `neo4j`                  | Neo4j username            |
| `NEO4J_PASSWORD` | `password`               | Neo4j password            |

Notes:

- If your Neo4j runs in the Compose stack and you connect to it **from your host**, use
  the mapped Bolt port: `bolt://localhost:8687` (not 7687, and not the internal service
  name `neo4j-v2`, which only resolves inside the Docker network).
- If you keep credentials in a `.env` file, save it with **LF (Unix) line endings**. A
  trailing `\r` from a Windows editor becomes part of the value (e.g. `password\r`) and
  causes authentication failures that are hard to spot.
