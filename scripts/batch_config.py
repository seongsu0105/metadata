"""
배치 스크립트: 경로, 환경, 워커 프롬프트 정본.

파인튜닝 라벨 (finetune_dataset.jsonl)
----------------------------------------
- instruction: ``METADATA_SYSTEM`` (ollama-api ``app.prompts.summary`` 와 동일).
- input: Ollama에 보낸 user 문자열 (= ``metadata_user_prompt`` 스니펫 포함).
- output: **최종 저장용 4줄** (``format_metadata_block``). LLM raw 출력이 아니라
  R1–R11 후처리·bc_id 교정·규칙 기반 ``extract_summary`` 폴백 등을 거친 값.
  순수 모델 출력 모방 학습이 목적이면 별도 플래그·필드로 raw를 남기도록 확장할 것.

입력 텍스트
-----------
운영 워커는 보통 파이프라인 요약을 「대상 텍스트」로 넣지만, 본 배치는
기본적으로 **PDF 텍스트 추출** 결과를 넣는다. 레이아웃 복잡·스캔 PDF는 품질 한계가 있다.
"""

from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_ROOT = os.path.dirname(SCRIPT_DIR)
_REPO_SIBLING_ROOT = os.path.dirname(METADATA_ROOT)
_env_api = os.environ.get("OLLAMA_API_ROOT", "").strip()
OLLAMA_API_ROOT = (
    os.path.normpath(_env_api)
    if _env_api
    else os.path.normpath(os.path.join(_REPO_SIBLING_ROOT, "ollama-api"))
)
if os.path.isdir(OLLAMA_API_ROOT) and OLLAMA_API_ROOT not in sys.path:
    sys.path.insert(0, OLLAMA_API_ROOT)

try:
    from app.prompts.summary import (
        METADATA_SYSTEM,
        build_big_categories_block,
        metadata_user_prompt,
    )
except ImportError as e:
    raise ImportError(
        "ollama-api의 app.prompts.summary를 불러올 수 없습니다. "
        f"OLLAMA_API_ROOT 환경 변수로 경로를 지정하거나, 형제 폴더에 두세요 (기대: {OLLAMA_API_ROOT})."
    ) from e

# PDF 입력·JSONL 출력: cwd가 아닌 metadata 루트 기준 (환경 변수로 덮어쓰기)
PDF_DIR = os.environ.get("METADATA_PDF_DIR", "").strip() or os.path.join(
    METADATA_ROOT, "pdfdata"
)
OUTPUT_DIR = os.environ.get("METADATA_OUTPUT_DIR", "").strip() or os.path.join(
    METADATA_ROOT, "out"
)
METADATA_JSONL = os.path.join(OUTPUT_DIR, "metadata.jsonl")
FINETUNE_JSONL = os.path.join(OUTPUT_DIR, "finetune_dataset.jsonl")

SANITIZE_OVERRIDES_JSON = os.path.join(SCRIPT_DIR, "metadata_sanitize_overrides.json")

WRITE_FINETUNE = True

DEFAULT_METADATA_MAX_SUMMARY_CHARS = 12000

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_NUM_PREDICT = 1024
OLLAMA_CONNECT_TIMEOUT = 30
OLLAMA_READ_TIMEOUT = 900
OLLAMA_HTTP_RETRIES = 2
METADATA_QUALITY_RETRIES = 2

#   METADATA_FAST=1
#   METADATA_MAX_SUMMARY_CHARS=n
#   METADATA_MAX_PDFS=N
#   METADATA_PARALLEL_WORKERS=k
#   METADATA_PDF_DIR, METADATA_OUTPUT_DIR
#   OLLAMA_API_ROOT, OLLAMA_URL, OLLAMA_MODEL, …


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "y", "on")


def metadata_fast_enabled() -> bool:
    return _env_truthy("METADATA_FAST")


def effective_quality_retries() -> int:
    return 0 if metadata_fast_enabled() else METADATA_QUALITY_RETRIES


def effective_ollama_num_predict() -> int:
    if metadata_fast_enabled():
        return min(OLLAMA_NUM_PREDICT, 512)
    return OLLAMA_NUM_PREDICT


def effective_metadata_max_chars() -> int:
    raw = os.environ.get("METADATA_MAX_SUMMARY_CHARS", "").strip()
    if raw:
        try:
            return max(400, int(raw))
        except ValueError:
            pass
    if metadata_fast_enabled():
        return min(4000, DEFAULT_METADATA_MAX_SUMMARY_CHARS)
    return DEFAULT_METADATA_MAX_SUMMARY_CHARS


def max_pdf_files_env() -> int:
    raw = os.environ.get("METADATA_MAX_PDFS", "").strip()
    if not raw:
        return 0
    try:
        n = int(raw)
    except ValueError:
        return 0
    return max(0, n)


def parallel_workers_env() -> int:
    raw = os.environ.get("METADATA_PARALLEL_WORKERS", "").strip()
    if not raw:
        return 1
    try:
        k = int(raw)
    except ValueError:
        return 1
    return max(1, k)


# 인턴십 DB big_ctgrs 와 id·이름을 맞출 것
DB_BIG_CATEGORIES = [
    {"id": 1, "name": "정치/행정"},
    {"id": 2, "name": "경제/산업"},
    {"id": 3, "name": "사회/복지"},
    {"id": 4, "name": "법사/안전"},
    {"id": 5, "name": "교육/문화"},
    {"id": 6, "name": "과학/기술"},
    {"id": 7, "name": "국방/외교"},
    {"id": 8, "name": "환경/에너지"},
    {"id": 9, "name": "농림/축산"},
    {"id": 10, "name": "국토/교통"},
]

VALID_BC_IDS = {int(c["id"]) for c in DB_BIG_CATEGORIES}
