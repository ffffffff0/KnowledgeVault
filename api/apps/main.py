from fastapi import FastAPI

from api.apps.routes import api, kb, document, llm

app = FastAPI(title="RAG API", version="1.0.0")
app.include_router(api.router)
app.include_router(kb.router)
app.include_router(document.router)
app.include_router(llm.router)
