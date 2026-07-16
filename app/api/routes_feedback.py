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


class NegativeTurn(BaseModel):
    feedback_id: str
    conversation_id: str
    turn_idx: int
    comment: str | None
    created_at: float


class NegativeFeedbackResponse(BaseModel):
    items: list[NegativeTurn]
    total: int


@router.get("/negative", response_model=NegativeFeedbackResponse)
def get_negative_feedback(limit: int = 50):
    """Return recent thumbs-down turns for review."""
    with sqlite3.connect(_db()) as con:
        rows = con.execute(
            "SELECT id, conversation_id, turn_idx, comment, created_at FROM feedback "
            "WHERE rating='down' ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    items = [
        NegativeTurn(
            feedback_id=r[0],
            conversation_id=r[1],
            turn_idx=r[2],
            comment=r[3],
            created_at=r[4],
        )
        for r in rows
    ]
    return NegativeFeedbackResponse(items=items, total=len(items))


class PromoteResult(BaseModel):
    status: str
    golden_file: str
    entry: dict


@router.post("/promote/{feedback_id}", response_model=PromoteResult)
def promote_to_golden(feedback_id: str):
    """Promote a thumbs-down turn to the golden eval set.

    Appends the turn's user message + assistant response to
    data/eval/golden_feedback.jsonl for later human review and labelling.
    """
    import json as _json

    with sqlite3.connect(_db()) as con:
        row = con.execute(
            "SELECT conversation_id, turn_idx, comment FROM feedback WHERE id=? AND rating='down'",
            (feedback_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Feedback entry not found or not a thumbs-down.")

    conversation_id, turn_idx, comment = row

    user_message: str | None = None
    assistant_message: str | None = None
    try:
        from app.agent.conversation import ConversationStore
        cs = ConversationStore()
        conv = cs.get_or_create(conversation_id)
        turns = conv.turns
        if 0 <= turn_idx < len(turns):
            t = turns[turn_idx]
            if t.role == "assistant":
                assistant_message = t.content
                if turn_idx > 0 and turns[turn_idx - 1].role == "user":
                    user_message = turns[turn_idx - 1].content
    except Exception:
        pass

    from app.core.config import settings
    golden_dir = settings.eval_path
    golden_dir.mkdir(parents=True, exist_ok=True)
    golden_file = golden_dir / "golden_feedback.jsonl"

    entry = {
        "feedback_id": feedback_id,
        "conversation_id": conversation_id,
        "turn_idx": turn_idx,
        "user_message": user_message,
        "assistant_message": assistant_message,
        "comment": comment,
        "promoted_at": time.time(),
        "expected_tool": None,  # human fills this in after review
    }
    with open(golden_file, "a") as f:
        f.write(_json.dumps(entry) + "\n")

    return PromoteResult(status="promoted", golden_file=str(golden_file), entry=entry)
