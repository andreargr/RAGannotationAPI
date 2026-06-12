# RAGannotationAPI — API Server

This directory contains the FastAPI application that exposes the RAG annotation endpoints.

> **Note:** If you are running the full system via Docker Compose, refer to the [main README](../README.md). The instructions below are intended for manual or development setups only.

---

## Prerequisites

Before running the API, the following must be completed:

1. Neo4j must be installed and running (version ≥ 5.26.6)
2. Ontology files must be placed in `../embeddings/ontologies/`
3. Embeddings must have been generated and stored in Neo4j by running:

```bash
python ../embeddings/get_store_embeddings_configfile.py
```

> The API depends on these embeddings to perform semantic search. Without them, annotation endpoints will not return results.

---

## Installation

From the project root:

```bash
pip install -r requirements.txt
```

---

## Running the API

Navigate to this directory:

```bash
cd api
```

Start the server:

```bash
uvicorn main:app --reload
```

The API will be available at:

```
http://127.0.0.1:8000
```

Interactive documentation (Swagger UI) is available at:

```
http://127.0.0.1:8000/docs
```

The `--reload` flag enables automatic server restart on code changes, which is recommended during development.

---

## Environment variables

The following environment variables configure the Neo4j connection. When running with Docker Compose, these are set automatically from the `.env` file in the project root. For manual execution, you can export them in your shell or create a local `.env` file and load it before starting the server:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
```

| Variable | Default | Description |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
