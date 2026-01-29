from rdflib import Graph, RDF, RDFS, OWL, URIRef
from rdflib.namespace import SKOS
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

NEO4J_SUMMARY_FOLDER="./summary"
NEO4J_EMBEDDING_FOLDER="embeddings"

def get_class_name(cls):
    """
    Returns the first label in the class, or a default name if there are no labels.
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

    # def store_class(self, ontology_id, cls, embedding):
    #     """
    #     Guarda un nodo Class y lo relaciona con la Ontology correspondiente.
    #     """
    #     query = """
    #     MERGE (c:Class {id: $id})
    #     SET c.name = $name,
    #         c.labels = $labels,
    #         c.comment = $comment,
    #         c.synonyms = $synonyms,
    #         c.objectProperties = $objectProperties,
    #         c.dataProperties = $dataProperties,
    #         c.embedding = $embedding
    #     WITH c
    #     MATCH (o:Ontology {id: $ontology_id})
    #     MERGE (o)-[:HAS_CLASS]->(c)
    #     """
    #     name = get_class_name(cls)
    #     self.driver.execute_query(
    #         query,
    #         id=cls["id"],
    #         name = name,
    #         labels=cls.get("label", []),
    #         comment=cls.get("comment", ""),
    #         synonyms=cls.get("synonyms", []),
    #         objectProperties=cls.get("objectProperties", []),
    #         dataProperties=cls.get("dataProperties", []),
    #         embedding=embedding,
    #         ontology_id=ontology_id
    #     )


def get_ontology_list(directory="../core-ontologies-siemensenergy"):
    dir_path = Path(directory)
    extensions = ["*.ttl", "*.owl", "*.rdf", "*.xml"]

    ontologies = []
    for ext in extensions:
        for file in dir_path.glob(ext):
            ontologies.append({
                "id": file.stem,
                "filename": file.name,
                "path": str(file),
                "content": file.read_text(encoding="utf-8", errors="ignore")
            })

    return ontologies

def get_graph(dataset):
    g = Graph()
    try:
        g.parse(dataset["path"]) #Parse by file path, for format auto-detection
    except Exception as e:
        print(f"⚠️ Error parsing {dataset['path']}: {e}")
        return None
    return g

def get_preferred_name(g, uri):
    # 1. rdfs:label
    for o in g.objects(uri, RDFS.label):
        return str(o)

    # 2. QName
    try:
        return g.qname(uri)
    except Exception:
        return str(uri)


def get_all_labels(g, uri):
    return [str(o) for o in g.objects(uri, RDFS.label)]


def get_comment(g, uri):
    for o in g.objects(uri, RDFS.comment):
        return str(o)
    return ""


def get_synonyms(g, uri):
    synonyms = []
    for p in [SKOS.altLabel, SKOS.hiddenLabel]:
        synonyms.extend(str(o) for o in g.objects(uri, p))
    return synonyms


def get_individuals_of_class(g, class_uri):
    individuals = []
    for ind in g.subjects(RDF.type, class_uri):
        if isinstance(ind, URIRef):
            individuals.append({
                "iri": str(ind),
                "label": get_preferred_name(g, ind)
            })
    return individuals


def extract_class_structure(g, class_uri):
    super_classes = [
        get_preferred_name(g, o)
        for o in g.objects(class_uri, RDFS.subClassOf)
        if isinstance(o, URIRef)
    ]

    sub_classes = [
        get_preferred_name(g, s)
        for s in g.subjects(RDFS.subClassOf, class_uri)
        if isinstance(s, URIRef)
    ]

    object_properties = [
        get_preferred_name(g, p)
        for p in g.subjects(RDF.type, OWL.ObjectProperty)
        if class_uri in g.objects(p, RDFS.domain)
    ]

    data_properties = [
        get_preferred_name(g, p)
        for p in g.subjects(RDF.type, OWL.DatatypeProperty)
        if class_uri in g.objects(p, RDFS.domain)
    ]

    equivalent_classes = [
        get_preferred_name(g, o)
        for o in g.objects(class_uri, OWL.equivalentClass)
        if isinstance(o, URIRef)
    ]

    disjoint_classes = [
        get_preferred_name(g, o)
        for o in g.objects(class_uri, OWL.disjointWith)
        if isinstance(o, URIRef)
    ]

    individuals = get_individuals_of_class(g, class_uri)

    return {
        "id": str(class_uri),
        "type": "Class",
        "label": get_all_labels(g, class_uri),
        "comment": get_comment(g, class_uri),
        "synonyms": get_synonyms(g, class_uri),
        "superClasses": super_classes,
        "subClasses": sub_classes,
        "objectProperties": object_properties,
        "dataProperties": data_properties,
        "equivalentClasses": equivalent_classes,
        "disjointWith": disjoint_classes,
        "individuals": individuals
    }


def extract_all_classes(g):
    classes = []

    for class_uri in g.subjects(RDF.type, OWL.Class):
        if not isinstance(class_uri, URIRef):
            continue
        classes.append(extract_class_structure(g, class_uri))

    return classes

def summary_to_text(classes_summary):
    texts = []

    for cls in classes_summary:
        text = f"Class {', '.join(cls.get('label', []))}. "

        if cls.get("comment"):
            text += f"Description: {cls['comment']}. "

        if cls.get("superClasses"):
            text += f"Superclasses: {', '.join(cls['superClasses'])}. "

        if cls.get("subClasses"):
            text += f"Subclasses: {', '.join(cls['subClasses'])}. "

        if cls.get("objectProperties"):
            text += f"Object properties: {', '.join(cls['objectProperties'])}. "

        if cls.get("dataProperties"):
            text += f"Data properties: {', '.join(cls['dataProperties'])}. "

        if cls.get("individuals"):
            individuals_text = ", ".join(
                f"{ind['label']} ({ind['iri']})"
                for ind in cls["individuals"]
            )
            text += f"Individuals: {individuals_text}. "

        texts.append(text)

    return "\n".join(texts)


def process_ontology (dataset, neo4j_url, neo4j_user, neo4j_pwd):
    graph = get_graph(dataset)
    id = dataset["id"] #take the name of the ontology as the ID
    manager = Neo4jManager(neo4j_url, neo4j_user, neo4j_pwd)
    try:
        classes_summary = extract_all_classes(graph)  # extract ontology summary
        summary_text = summary_to_text(classes_summary)  # convert it to text to improve the quality of embeddings

        summary_path = f"{NEO4J_SUMMARY_FOLDER}/{id}.json"

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(classes_summary, file, indent=2, ensure_ascii=False)

        embedding = manager.embedding_model.encode(summary_text,
                normalize_embeddings=True).tolist() # get embeddings

        manager.store_ontology(  # store embeddings for each of the ontologies
            ontology_id=id,
            filename=dataset["filename"],
            content=dataset["content"],
            summary=json.dumps(classes_summary, ensure_ascii=False),
            embedding=embedding
        )

        # for cls in classes_summary:
        #     #print("Processing class:", cls)
        #     class_text = summary_to_text([cls])  # genera texto solo de esa clase
        #     class_embedding = manager.embedding_model.encode(class_text).tolist()
        #     manager.store_class(ontology_id=id, cls=cls, embedding=class_embedding)

    finally:
        manager.close()


if __name__ == "__main__":
    datasets = get_ontology_list()
    print("Ontologías encontradas:", len(datasets))

    manager = Neo4jManager("bolt://localhost:7687", "neo4j", "password123")
    manager.create_ontology_vector_index()

    max_workers = os.cpu_count()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_ontology, dataset, "bolt://localhost:7687", "neo4j", "password123") for dataset in datasets]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error: {e}")



