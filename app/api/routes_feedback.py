from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

_DB: Path | None = None


def _db() -> Path:
    global _DB
    if _DB is None:
        from app.core.config import settings
        _DB = settings.data_path / "conversations.db"
    return _DB


def _init_table() -> None:
    with sqlite3.connect(_db()) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                turn_idx INTEGER NOT NULL,
                rating TEXT NOT NULL,
                comment TEXT,
                created_at REAL NOT NULL
            )
        """)


_init_table()


class FeedbackRequest(BaseModel):
    conversation_id: str
    turn_idx: int
    rating: str
    comment: str | None = None


class FeedbackResponse(BaseModel):
    id: str
    status: str = "ok"


class FeedbackStats(BaseModel):
    total: int
    up: int
    down: int
    up_rate: float


@router.post("", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest):
    if req.rating not in ("up", "down"):
        raise HTTPException(status_code=422, detail="rating must be 'up' or 'down'")
    fid = str(uuid.uuid4())
    with sqlite3.connect(_db()) as con:
        con.execute(
            "INSERT OR REPLACE INTO feedback (id, conversation_id, turn_idx, rating, comment, created_at) VALUES (?,?,?,?,?,?)",
            (fid, req.conversation_id, req.turn_idx, req.rating, req.comment, time.time()),
        )
    return FeedbackResponse(id=fid)


@router.get("/stats", response_model=FeedbackStats)
def feedback_stats():
    with sqlite3.connect(_db()) as con:
        row = con.execute("SELECT COUNT(*), SUM(rating='up'), SUM(rating='down') FROM feedback").fetchone()
    total, up, down = int(row[0]), int(row[1] or 0), int(row[2] or 0)
    return FeedbackStats(total=total, up=up, down=down, up_rate=round(up / total, 3) if total else 0.0)
