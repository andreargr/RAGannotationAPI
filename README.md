
# OntonNG RAG annotation API

## ⚠️ Requisitos previos

Antes de ejecutar esta API, es **imprescindible** seguir los pasos de procesamiento de embeddings descritos en el proyecto de ontologías:

[Procesamiento de Ontologías y Almacenamiento de Embeddings en Neo4j](./embeddings/README.md)

En resumen:

1. Neo4j debe estar instalado y en ejecución (versión ≥ 5.26.6)  
2. Las ontologías deben colocarse en `/embeddings/ontologies/`  
3. Ejecutar `python get_store_embeddings.py` para generar y almacenar los embeddings en Neo4j  

> La API depende de estos embeddings para poder realizar búsquedas semánticas.

---

## 📦 Instalación de dependencias

Desde la raíz del proyecto o desde el directorio correspondiente:

```bash
pip install -r requirements.txt
````

## 🚀 Ejecución de la API

Para ejecutar la API correctamente, **debes situarte en el directorio `/api`**:

```bash
cd api
```

Luego ejecuta el servidor con el siguiente comando:

```bash
uvicorn main:app --reload
```

La API estará disponible por defecto en:

```
http://127.0.0.1:8000
```

El parámetro `--reload` habilita la recarga automática del servidor cuando se detectan cambios en el código, ideal para desarrollo.
