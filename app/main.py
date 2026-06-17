from __future__ import annotations

from fastapi import FastAPI

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.storage import ensure_data_paths

from app.api.routes_health import router as health_router
from app.api.routes_uploads import router as uploads_router
from app.api.routes_chat import router as chat_router
from app.api.routes_datasets import router as datasets_router
from app.api.routes_models import router as models_router
from app.api.routes_root import router as root_router


setup_logging()

app = FastAPI(title=settings.app_name)


@app.on_event("startup")
def _startup() -> None:
    ensure_data_paths()


app.include_router(root_router)
app.include_router(health_router)
app.include_router(uploads_router, prefix="/uploads", tags=["uploads"])
app.include_router(datasets_router, prefix="/datasets", tags=["datasets"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(models_router, prefix="/models", tags=["models"])
