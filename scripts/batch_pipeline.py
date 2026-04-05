from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from batch_config import (
    DB_BIG_CATEGORIES,
    FINETUNE_JSONL,
    METADATA_JSONL,
    METADATA_SYSTEM,
    OUTPUT_DIR,
    PDF_DIR,
    WRITE_FINETUNE,
    build_big_categories_block,
    max_pdf_files_env,
    parallel_workers_env,
)
from batch_llm import dataset_user_prompt
from batch_pdf import extract_pdf_text
from batch_record import build_record_for_pdf


def _process_one_pdf_file(
    file: str, pdf_dir: str, bc_block: str
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    """단일 PDF → (파일명, 메타 row 또는 None, 파인튜닝 row 또는 None). 병렬 워커에서도 사용."""
    try:
        path = os.path.join(pdf_dir, file)
        text = extract_pdf_text(path)

        if len(text) < 100:
            return file, None, None

        meta_row, output_block = build_record_for_pdf(file, text, bc_block)
        ft: dict[str, Any] | None = None
        if WRITE_FINETUNE:
            ft = {
                "instruction": METADATA_SYSTEM,
                "input": dataset_user_prompt(text, bc_block),
                "output": output_block,
            }
        return file, meta_row, ft
    except Exception as e:
        print(f"에러: {file}, {e}")
        return file, None, None


def _process_one_pdf_file_star(
    args: tuple[str, str, str],
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    return _process_one_pdf_file(*args)


def process_all() -> None:
    pdf_dir = PDF_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    meta_path = METADATA_JSONL
    meta_rows: list[dict[str, Any]] = []
    finetune_rows: list[dict[str, Any]] = []

    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(
            f"PDF 폴더가 없습니다: {pdf_dir} (METADATA_PDF_DIR 또는 metadata/pdfdata 를 확인하세요)"
        )

    bc_block = build_big_categories_block(DB_BIG_CATEGORIES)

    files = sorted(f for f in os.listdir(pdf_dir) if f.endswith(".pdf"))
    cap = max_pdf_files_env()
    if cap > 0:
        files = files[:cap]

    workers = parallel_workers_env()
    if workers == 1:
        for file in files:
            file_, meta_row, ft = _process_one_pdf_file(file, pdf_dir, bc_block)
            if meta_row is not None:
                meta_rows.append(meta_row)
                print(f"완료: {file_}")
            if ft is not None:
                finetune_rows.append(ft)
    else:
        print(
            "참고: 병렬 시 Ollama에 동시 요청이 들어갑니다. GPU 1대·VRAM 여유 없으면 "
            "METADATA_PARALLEL_WORKERS=1 이 더 빠를 수 있습니다."
        )
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = ex.map(
                _process_one_pdf_file_star,
                [(f, pdf_dir, bc_block) for f in files],
            )
        for file_, meta_row, ft in results:
            if meta_row is not None:
                meta_rows.append(meta_row)
                print(f"완료: {file_}")
            if ft is not None:
                finetune_rows.append(ft)

    with open(meta_path, "w", encoding="utf-8") as f:
        for r in meta_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if WRITE_FINETUNE and finetune_rows:
        with open(FINETUNE_JSONL, "w", encoding="utf-8") as f:
            for r in finetune_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"\nDB 메타데이터: {meta_path} ({len(meta_rows)}건)"
        + (
            f" | 학습용: {FINETUNE_JSONL} ({len(finetune_rows)}건)"
            if WRITE_FINETUNE
            else ""
        )
    )
