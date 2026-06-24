from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.storage import ensure_data_paths

from app.api.routes_health import router as health_router
from app.api.routes_uploads import router as uploads_router
from app.api.routes_chat import router as chat_router
from app.api.routes_corpus import router as corpus_router
from app.api.routes_datasets import router as datasets_router
from app.api.routes_experiments import router as experiments_router
from app.api.routes_models import router as models_router
from app.api.routes_root import router as root_router
from app.api.routes_training import router as training_router


setup_logging()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    ensure_data_paths()


app.include_router(root_router)
app.include_router(health_router)
app.include_router(uploads_router, prefix="/uploads", tags=["uploads"])
app.include_router(datasets_router, prefix="/datasets", tags=["datasets"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(models_router, prefix="/models", tags=["models"])
app.include_router(experiments_router, prefix="/experiments", tags=["experiments"])
app.include_router(training_router, prefix="/training", tags=["training"])
app.include_router(corpus_router, prefix="/corpus", tags=["corpus"])
