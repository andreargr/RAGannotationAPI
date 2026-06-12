# RAG annotation API

A REST API for semantic annotation of biomedical samples using Retrieval-Augmented Generation (RAG) over ontologies stored in a Neo4j vector database. The system retrieves ontology classes semantically similar to a given input label and uses a language model to assign the most appropriate annotation.

---

## Requirements

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)

No additional local dependencies are required. All services run inside containers.

---

## Project structure

```
RAGannotationAPI/
├── api/                  # FastAPI application
│   ├── Dockerfile
│   ├── main.py
│   └── README.md         # Manual execution instructions
├── embeddings/           # Embedding generation scripts and ontologies
│   ├── ontologies/       # Place ontology files here (.ttl, .owl, .rdf, .xml)
│   ├── get_store_embeddings.py
│   └── README.md         # Manual execution instructions
├── neo4j/                # Neo4j data volume (auto-generated)
├── docker-compose.yml
└── requirements.txt
```

---

## Quick start

### 1. Add ontologies

Place the ontology files to be indexed in `embeddings/ontologies/`. Supported formats: `.ttl`, `.owl`, `.rdf`, `.xml`.

### 2. Start the services

```bash
docker compose up -d
```

This starts two services:

| Service | Description | Ports |
|---|---|---|
| `api` | FastAPI REST API | `8000` |
| `neo4j` | Neo4j graph database | `7474` (browser), `7687` (Bolt) |

### 3. Load embeddings into Neo4j

This step must be run once before the API can serve requests. It parses the ontologies, generates semantic embeddings, and stores them in Neo4j.

```bash
docker compose exec api sh -c "cd /embeddings && python get_store_embeddings.py"
```

> ⚠️ Make sure the Neo4j container is fully started before running this command. You can verify it at `http://localhost:7474`.

### 4. Access the API

The API is available at:

```
http://localhost:8000
```

Interactive API documentation (Swagger UI) is available at:

```
http://localhost:8000/docs
```

---

## Stopping the services

```bash
docker compose down
```

To also remove the Neo4j data volume:

```bash
docker compose down -v
```

---

## Manual execution (without Docker)

For local development without Docker, refer to the individual component documentation:

- [Embedding generation](./embeddings/README.md)
- [API server](./api/README.md)
