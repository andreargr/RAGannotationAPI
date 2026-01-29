from fastapi import FastAPI
from ontong_rag.router_ontong_rag import router as ontology_router
from annotation.router_annotation import router as annotation_router

app = FastAPI(title="ontong-rag", docs_url="/")


app.include_router(ontology_router, prefix="/ontology", tags=["Ontology Similarity"])
app.include_router(annotation_router, prefix="/annotation", tags=["Annotation"])

