from __future__ import annotations

import re


def parse_llm_metadata_block(raw: str) -> dict[str, str]:
    """모델 텍스트에서 title/bc_id/sc_keyword/tl_summary 추출 (전각 콜론·줄 단위)."""
    out: dict[str, str] = {}
    if not raw:
        return out
    text = raw.replace("\ufeff", "").replace("\uff1a", ":")
    key_pat = re.compile(
        r"^\s*(title|bc_id|sc_keyword|tl_summary)\s*:\s*(.*)$",
        re.IGNORECASE,
    )
    last_key: str | None = None
    for line in text.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        m = key_pat.match(line)
        if m:
            k = m.group(1).lower()
            out[k] = m.group(2).strip()
            last_key = k
            continue
        if last_key == "tl_summary" and line.strip():
            out["tl_summary"] = (out.get("tl_summary", "") + " " + line.strip()).strip()
            continue
        if ": " in line:
            k, _, v = line.partition(": ")
            key = k.strip()
            if key in ("title", "bc_id", "sc_keyword", "tl_summary") and key not in out:
                out[key] = v.strip()
                last_key = key
    return out
