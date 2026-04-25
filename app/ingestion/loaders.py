from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pdfplumber
from PIL import Image

import pytesseract
from app.core.config import settings


@dataclass
class Ingested:
    kind: str  # "table" | "text"
    payload: Any
    notes: list[str]


def load_csv(content: bytes, filename: str) -> Ingested:
    bio = io.BytesIO(content)
    df = pd.read_csv(bio)
    return Ingested(kind="table", payload=df, notes=[f"Loaded CSV: {filename}"])


def load_excel(content: bytes, filename: str) -> Ingested:
    bio = io.BytesIO(content)
    xls = pd.ExcelFile(bio)
    notes = [f"Loaded Excel: {filename}", f"Sheets: {xls.sheet_names}"]
    # Default: first sheet; you can extend to multi-sheet dataset-per-sheet
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    return Ingested(kind="table", payload=df, notes=notes)


def load_pdf_text(content: bytes, filename: str) -> Ingested:
    notes: list[str] = [f"Loaded PDF: {filename}"]
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        notes.append(f"Pages: {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages):
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(f"[page {i+1}]\n{t}")
    text = "\n\n".join(text_parts).strip()
    if not text:
        notes.append("No extractable text found (PDF may be scanned). Consider image OCR pipeline.")
    return Ingested(kind="text", payload=text, notes=notes)


def load_image_ocr(content: bytes, filename: str) -> Ingested:
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    img = Image.open(io.BytesIO(content)).convert("RGB")
    text = pytesseract.image_to_string(img)
    notes = [f"OCR image: {filename}", "OCR quality depends on image resolution and clarity."]
    return Ingested(kind="text", payload=text.strip(), notes=notes)
