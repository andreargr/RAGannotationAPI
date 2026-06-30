# RAG Annotation API

A REST API for semantic annotation of samples using Retrieval-Augmented Generation (RAG)
over ontologies stored in a **Neo4j** vector database. The system retrieves ontology
classes semantically similar to a given input and uses the result to assign the most
appropriate annotation.

---

## Requirements

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)

No additional local dependencies are required: all services run inside containers.

---

## Project structure

```
RAGannotationAPI/
├── api/                       # FastAPI application
│   ├── Dockerfile
│   ├── main.py
│   └── README.md              # Manual / development execution
├── embeddings/                # Embedding generation scripts and ontologies
│   ├── ontologies/            # Place ontology files here (.ttl, .owl, .rdf, .xml)
│   ├── get_store_embeddings_classemb.py
│   ├── config.json            # RDF property mappings (labels/synonyms/descriptions)
│   └── README.md              # Manual / development execution
├── neo4j-data-cmd/            # Neo4j data volume (auto-generated, bind mount)
├── .env                       # Neo4j credentials (see below)
└── docker-compose.yml
```

---

## 1. Configure credentials (`.env`)

Create a `.env` file in the project root (next to `docker-compose.yml`) with the Neo4j
credentials. Docker Compose reads it automatically and injects the values into both
containers:

```
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_URI=bolt://neo4j-v2:7687
```

Notes:

- `NEO4J_URI` uses the **service name** `neo4j-v2`, which is the host that resolves
  inside the Docker network. Keep it as is when running through Compose.
- Save the file with **LF (Unix) line endings**, not CRLF. A trailing `\r` from a
  Windows editor becomes an invisible part of the value (e.g. `password\r`) and breaks
  authentication. In your editor select `LF`, or run `sed -i 's/\r$//' .env`.
- The password is only applied when the Neo4j database is **first created**. If you
  change it later, you must recreate the database volume (see *Resetting* below) for the
  new password to take effect.

---

## 2. Add ontologies

Place the ontology files to be indexed in `embeddings/ontologies/`.
Supported formats: `.ttl`, `.owl`, `.rdf`, `.xml`. Each file is processed as an
independent ontology.

Optionally, adjust `embeddings/config.json` to map which RDF properties are used as
labels, synonyms, and descriptions (see `embeddings/README.md` for details).

---

## 3. Start the services

```bash
docker compose up -d
```

This starts two services:

| Service     | Description           | Host ports                                   |
|-------------|-----------------------|----------------------------------------------|
| `api-oskar` | FastAPI REST API      | `9000` → container `8000`                    |
| `neo4j-v2`  | Neo4j graph database  | `8484` → `7474` (browser), `8687` → `7687` (Bolt) |

You can confirm Neo4j is up before continuing:

```bash
docker compose logs -f neo4j-v2     # wait for "Started.", then Ctrl+C
```

The Neo4j browser is available at `http://localhost:8484`.

---

## 4. Load embeddings into Neo4j

This step must be run **once** before the API can serve requests. It parses the
ontologies, generates the embeddings, and stores them in Neo4j. Run it from inside the
API container (where `neo4j-v2` resolves and `embeddings/` is mounted at `/embeddings`):

```bash
docker compose exec api-oskar sh -c "cd /embeddings && python get_store_embeddings_classemb.py"
```

To wipe the existing ontology data and recompute everything from scratch:

```bash
docker compose exec api-oskar sh -c "cd /embeddings && python get_store_embeddings_classemb.py --reset"
```

> Re-running without `--reset` skips ontologies whose content has not changed since the
> last run (based on a hash of the generated summary).

---

## 5. Access the API

The API is available at:

```
http://localhost:9000
```

Interactive documentation (Swagger UI):

```
http://localhost:9000/docs
```

---

## Stopping the services

```bash
docker compose down
```

The Neo4j data lives in the bind-mounted folder `./neo4j-data-cmd`, so it **persists**
across `docker compose down` (and is not removed by `-v`).

---

## Resetting Neo4j from scratch

Use this when you need a clean database (for example, after changing the password):

```bash
docker compose down
sudo rm -rf ./neo4j-data-cmd
docker compose up -d
```

Then re-run the embedding load (step 4). Alternatively, if you only want to clear the
ontology data without touching the container, use the `--reset` flag described in step 4.

---

## Troubleshooting

- **API can't authenticate against Neo4j.** Almost always a credentials mismatch.
  Verify the value actually reaching the container has no hidden characters:

  ```bash
  docker compose exec api-oskar sh -c "printenv NEO4J_PASSWORD | cat -A"
  ```

  It must print `password$`, not `password^M$` (the `^M` means CRLF in your `.env`).
  Also remember the password is fixed at database creation: if it was first created with
  a different one, reset the volume (see above).

- **`Failed to DNS resolve address neo4j-v2:7687`.** You are running outside the Docker
  network (e.g. the loader from your host). Inside Docker use `neo4j-v2:7687`; from your
  host use `bolt://localhost:8687` instead.

- **Empty results from the API.** The embeddings step (step 4) has not been run, or it
  ran against a different/empty database.

- **Check what Compose resolved** (useful to confirm the `.env` was picked up):

  ```bash
  docker compose config
  ```

---

## Manual execution (without Docker)

For local development without Docker, refer to the component documentation:

- [Embedding generation](./embeddings/README.md)
- [API server](./api/README.md)
