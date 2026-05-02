from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.store import router as store_router

app = FastAPI(
    title="TCC Chatbot API",
    description="API para gerenciar stores do Gemini File Search",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(store_router, prefix="/store", tags=["Store"])


@app.get("/")
def root():
    return {
        "status": "ok",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": {
            "stores": "/store/listar",
            "indexar": "/store/indexar",
            "chat": "/store/chat",
        },
    }