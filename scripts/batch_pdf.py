"""
PDF → 평문. PyPDF2 텍스트 레이어 추출만 사용 (스캔 PDF·복잡 레이아웃은 빈약할 수 있음).
운영 파이프라인급 품질이 필요하면 OCR 결과 텍스트를 별도 파이프라인에서 넣도록 확장.
"""

from __future__ import annotations

import re

from PyPDF2 import PdfReader


def clean_text(text: str) -> str:
    text = re.sub(r"[\n\t\r]+", " ", text)
    text = re.sub(r"([가-힣])\s{2,}([가-힣])", r"\1\2", text)
    text = re.sub(r"([가-힣])\s+([음함임됨])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    parts = [p.extract_text() or "" for p in reader.pages if p.extract_text()]
    return clean_text(" ".join(parts))
