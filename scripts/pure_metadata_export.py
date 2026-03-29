"""
DB용 순수 메타데이터 JSONL 변환·검증.

- main.py 기본 산출물은 metadata.jsonl (이미 순수 필드만 있음).
- 예전 dataset.jsonl / finetune_dataset.jsonl(output 블록)에서 DB 스키마만 뽑을 때 사용.

사용 (metadata 폴더에서):
  python scripts/pure_metadata_export.py
  python scripts/pure_metadata_export.py --in finetune_dataset.jsonl --out import.jsonl
  python scripts/pure_metadata_export.py --in dataset.jsonl  # 레거시
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_metadata_block(block: str) -> dict[str, str]:
    """'title: ...\\nbc_id: ...' 형식 문자열을 필드 dict로 분해."""
    out: dict[str, str] = {}
    for line in (block or "").splitlines():
        if ": " not in line:
            continue
        key, val = line.split(": ", 1)
        k = key.strip()
        if k in ("title", "bc_id", "sc_keyword", "tl_summary"):
            out[k] = val.strip()
    return out


def row_to_db_row(row: dict) -> dict | None:
    """
    한 줄 JSON → DB import용 dict.
    - 이미 metadata 행이면 그대로 정규화
    - 학습용(finetune)이면 output에서 파싱
    """
    if row.get("output") and isinstance(row["output"], str):
        meta = parse_metadata_block(row["output"])
        if not meta.get("title"):
            return None
        rec = {
            "title": meta.get("title", ""),
            "bc_id": meta.get("bc_id", ""),
            "sc_keyword": meta.get("sc_keyword", ""),
            "tl_summary": meta.get("tl_summary", ""),
        }
        if row.get("source_pdf"):
            rec["source_pdf"] = row["source_pdf"]
        return rec

    if row.get("title"):
        rec = {
            "title": row.get("title", ""),
            "bc_id": row.get("bc_id", ""),
            "sc_keyword": row.get("sc_keyword", ""),
            "tl_summary": row.get("tl_summary", ""),
        }
        if row.get("source_pdf"):
            rec["source_pdf"] = row["source_pdf"]
        return rec

    return None


def export_from_jsonl(dataset_path: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(dataset_path, encoding="utf-8") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            db_row = row_to_db_row(row)
            if db_row is None:
                continue
            fout.write(json.dumps(db_row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="JSONL → DB용 메타데이터 JSONL (metadata 행 또는 output 파싱)"
    )
    p.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=None,
        help="입력 JSONL (기본: metadata.jsonl 있으면 그것, 없으면 dataset.jsonl)",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=root / "metadata_db.jsonl",
        help="출력 JSONL",
    )
    args = p.parse_args()
    in_path = args.in_path
    if in_path is None:
        meta = root / "metadata.jsonl"
        legacy = root / "dataset.jsonl"
        in_path = meta if meta.is_file() else legacy
    out_path: Path = args.out_path
    if not in_path.is_file():
        raise SystemExit(f"입력 파일이 없습니다: {in_path}")
    count = export_from_jsonl(in_path, out_path)
    print(f"쓰기 완료: {out_path} ({count}건) ← {in_path.name}")


if __name__ == "__main__":
    main()
