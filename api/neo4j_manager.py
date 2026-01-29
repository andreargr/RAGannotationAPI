from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import numpy as np
import json

def get_class_name(cls):
    """
    Devuelve el primer label de la clase, o un nombre por defecto si no tiene labels.
    """
    labels = cls.get("label")
    if labels and len(labels) > 0:
        return labels[0]
    return "UnnamedClass"

class Neo4jManager:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Load a embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Define authentication data
        self.auth_data = {
            'uri': uri,
            'database': 'neo4j',
            'user': user,
            'pwd': password
        }

    def close(self):
        self.driver.close()

    def create_ontology_vector_index(self):
        delete_index = """
        DROP INDEX `ontology-embeddings` IF EXISTS;
        """
        check_index_query = """
        SHOW INDEXES YIELD name
        WHERE name = 'ontology-embeddings'
        RETURN name
        """
        create_index_query = """
        CREATE VECTOR INDEX `ontology-embeddings`
        FOR (n:Ontology) ON (n.embedding)
        OPTIONS {indexConfig: {
          `vector.dimensions`: 384,
          `vector.similarity_function`: 'cosine'
        }};
        """

        with self.driver.session() as session:
            session.run(delete_index)
            result = session.run(check_index_query)
            if not result.single():
                session.run(create_index_query)
                print("Vector index 'ontology-embeddings' created successfully.")
            else:
                print("Vector index 'ontology-embeddings' already exists.")

    def store_ontology(self, ontology_id, filename, content, summary, embedding):
        query = """
        MERGE (o:Ontology {id: $id})
        SET o.filename = $filename,
            o.content = $content,
            o.summary = $summary,
            o.embedding = $embedding
        """
        self.driver.execute_query(
            query,
            id=ontology_id,
            filename=filename,
            content=content,
            summary=summary,
            embedding=embedding
        )

    def store_class(self, ontology_id, cls, embedding):
        """
        Guarda un nodo Class y lo relaciona con la Ontology correspondiente.
        """
        query = """
        MERGE (c:Class {id: $id})
        SET c.name = $name,
            c.labels = $labels,
            c.comment = $comment,
            c.synonyms = $synonyms,
            c.objectProperties = $objectProperties,
            c.dataProperties = $dataProperties,
            c.embedding = $embedding
        WITH c
        MATCH (o:Ontology {id: $ontology_id})
        MERGE (o)-[:HAS_CLASS]->(c)
        """
        name = get_class_name(cls)
        self.driver.execute_query(
            query,
            id=cls["id"],
            name = name,
            labels=cls.get("label", []),
            comment=cls.get("comment", ""),
            synonyms=cls.get("synonyms", []),
            objectProperties=cls.get("objectProperties", []),
            dataProperties=cls.get("dataProperties", []),
            embedding=embedding,
            ontology_id=ontology_id
        )

    # Similarity search by ontology embeddings
    def find_most_similar_ontology(self, input_text, top_k=10):
        lines = input_text.splitlines() #Divide el texto por saltos de línea (\n)
        lines = [line.strip() for line in lines if line.strip()] #Limpia líneas vacías y espacios
        embeddings = self.embedding_model.encode( #Generar embeddings por línea/término del input text
            lines,
            batch_size=512,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        avg_embedding = np.mean(embeddings, axis=0).tolist() #Combina todos los embeddings de líneas en un solo vector. Se obtiene una representación global del significado

        # Search the vector using vector search based on ontology embeddings
        query = """
        CALL db.index.vector.queryNodes(
          'ontology-embeddings',
          $top_k,
          $embedding
        )
        YIELD node, score
        RETURN
          node.id AS ontologyId,
          node.filename AS filename,
          score
        ORDER BY score DESC
        """
        #resultado ordenado de mayor a menor similitud
        with self.driver.session() as session:
            result = session.run(
                query,
                top_k=top_k,
                embedding=avg_embedding
            )
            return [record.data() for record in result]

    def get_ontology_summary(self, ontology_id):
        query = """
        MATCH (o:Ontology {id: $ontology_id})
        RETURN o.summary AS summary
        """

        with self.driver.session() as session:
            result = session.run(query, ontology_id=ontology_id)
            record = result.single()
            if record and record["summary"]:
                return json.loads(record["summary"])  # <-- JSON a lista de dicts
            return []


NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

manager = Neo4jManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)