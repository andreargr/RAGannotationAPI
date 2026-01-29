# Procesamiento de Ontologías y Almacenamiento de Embeddings en Neo4j

Este proyecto carga ontologías RDF/OWL, extrae su estructura (clases, propiedades, individuos), genera *embeddings* semánticos y los almacena en **Neo4j** utilizando un **índice vectorial** para búsquedas por similitud.

---

## Requisitos previos

### 1. Neo4j

Antes de ejecutar el código es **imprescindible que Neo4j esté instalado y en ejecución**.

* Neo4j **5.26.6** o superior (necesario para índices vectoriales)
* Acceso vía **Bolt** (por defecto `bolt://localhost:7687`)
* Usuario y contraseña válidos

Por defecto, el script usa:

```text
URI: bolt://localhost:7687
Usuario: neo4j
Contraseña: password123
```

> ⚠️ Ajusta estas credenciales en el código si tu configuración es diferente.

---

### 2. Entorno Python

Usar versión Python **3.10**.

Dependencias principales:

* `rdflib`
* `sentence-transformers`
* `neo4j`

Ejemplo de instalación:

```bash
pip install requirements.txt
```

---

## Estructura de directorios

El proyecto espera la siguiente estructura mínima:

```text
.
├── embeddings/
│   └── ontologies/
│       ├── example.owl
│       ├── example.ttl
│       └── example.rdf
└── README.md
```

### Directorios importantes

#### `/embeddings/ontologies/`

* **Aquí deben colocarse todas las ontologías** que se quieran procesar.
* Formatos soportados:

  * `.ttl`
  * `.owl`
  * `.rdf`
  * `.xml`

Cada archivo será tratado como una ontología independiente.

---

## Qué hace el script

1. Carga todas las ontologías desde `/embeddings/ontologies/`
2. Parsea los grafos RDF/OWL con **rdflib**
3. Extrae:

   * Clases (`owl:Class`)
   * Superclases y subclases
   * Propiedades de objeto y de datos
   * Individuos
   * Etiquetas, comentarios y sinónimos
4. Genera un resumen textual de cada ontología
5. Calcula *embeddings* usando:

```text
sentence-transformers/all-MiniLM-L6-v2
```

6. Crea (o recrea) un **índice vectorial en Neo4j**:

```text
ontology-embeddings
```

7. Almacena en Neo4j un nodo `:Ontology` por ontología con:

   * `id`
   * `filename`
   * `content`
   * `summary`
   * `embedding`

---

## Ejecución

Una vez:

* Neo4j esté levantado
* Las ontologías estén en `/embeddings/ontologies/`

Ejecuta:

```bash
python get_store_embeddings.py
```

El procesamiento se realiza en **paralelo**, utilizando todos los núcleos disponibles de la CPU.

---

## Notas adicionales

* Si una ontología no puede parsearse, se mostrará un aviso y se omitirá.
* El código incluye (comentada) la lógica para almacenar también clases individuales como nodos separados en Neo4j.
* Los embeddings se normalizan para mejorar la calidad de la similitud coseno.

---
