# =============================================================================
#   app\vllm\summary\postprocess.py
# =============================================================================
from __future__ import annotations

import re

from app.vllm.summary.metadata_source import (
    candidate_blob_for_keywords,
    extract_keyword_candidates_from_title,
    fallback_tl_summary_from_source,
    reconcile_title_for_metadata,
    scrub_sc_keyword_phrase,
    sc_keyword_parts_plausible,
    tl_summary_needs_source_fallback,
)

_PARTIAL_LINE_RE = re.compile(r"^\s*-\s*([^:]+)\s*:\s*(.+?)\s*$")
_EMPTY_TOKENS = {"없음", "해당 없음", "원문상 명시 없음", "미정", "n/a", "na"}
_PROMPT_LEAK_LINE_RE = re.compile(
    r"^\s*(?:\[청킹 좌표:|\[원문 일부:|주의할 점|목적·배경:|제도내용:|시행요소:|의안번호:|제목:)\b"
)
_META_KEY_PATTERNS: dict[str, re.Pattern[str]] = {
    "title": re.compile(r"^\s*(?:\d+\)\s*)?title\s*:\s*(.*)\s*$", re.IGNORECASE),
    "bc_id": re.compile(r"^\s*(?:\d+\)\s*)?bc_id\s*:\s*(.*)\s*$", re.IGNORECASE),
    "sc_keyword": re.compile(r"^\s*(?:\d+\)\s*)?sc_keyword\s*:\s*(.*)\s*$", re.IGNORECASE),
    "tl_summary": re.compile(r"^\s*(?:\d+\)\s*)?tl_summary\s*:\s*(.*)\s*$", re.IGNORECASE),
}
_TL_CONT_NOISE_LINE = re.compile(
    r"^(\s*#{1,6}\s*|세\s*줄\s*요약\s*$|\d+\.\s+\S.{0,100}$)"
)
_FINAL_SECTION_ORDER = (
    "식별정보",
    "핵심쟁점",
    "조문변경",
    "시행·부칙",
    "기타",
)
_SC_KEYWORD_STOPWORDS = {
    "요약",
    "법안",
    "제정안",
    "개정안",
    "대한민국",
    "정부",
    "위원회",
    "사건",
    "필요",
}
_GENERIC_TITLE_PREFIXES = (
    "요약",
    "법률안 요약",
    "국회 의안 통합 요약",
    "제정안",
    "개정안",
)


def _normalize_ws(s: str) -> str:
    return " ".join(str(s).strip().split())


def strip_metadata_title_suffixes(title: str) -> str:
    """LLM이 title 끝에 붙이는 `[bc_id]` 등 플레이스홀더를 제거한다."""
    t = (title or "").strip()
    tail_patterns = (
        r"\s*【\s*bc_id\s*】\s*$",
        r"\s*\[\s*bc_id\s*\]\s*$",
        r"\s*\(\s*bc_id\s*\)\s*$",
        r"\s+bc_id\s*$",
    )
    for _ in range(4):
        prev = t
        for p in tail_patterns:
            t = re.sub(p, "", t, flags=re.IGNORECASE).strip()
        if t == prev:
            break
    return t


def allowed_bc_ids_from_payload(big_categories: list[dict] | None) -> set[int] | None:
    """파이프라인에 넘어온 대분류 dict 목록에서 허용 id 집합. 없으면 None(검증 생략)."""
    if not big_categories:
        return None
    out: set[int] = set()
    for c in big_categories:
        cid = c.get("id")
        if cid is None:
            continue
        try:
            out.add(int(cid))
        except (TypeError, ValueError):
            continue
    return out if out else None


def finalize_metadata_line_output(
    normalized_four_lines: str,
    *,
    allowed_bc_ids: set[int] | None,
) -> str:
    """
    메타 4줄 문자열에 대해 title 꼬리 제거, bc_id를 허용 목록으로 검증(불일치 시 0).
    """
    key_vals: dict[str, str] = {}
    for ln in (normalized_four_lines or "").strip().splitlines():
        if ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        key_vals[k.strip()] = v.strip()
    title = strip_metadata_title_suffixes(key_vals.get("title", ""))
    if not title:
        title = "원문상 명시 없음"
    raw_bc = key_vals.get("bc_id", "")
    bc_digits = re.sub(r"[^\d]", "", raw_bc)
    if allowed_bc_ids:
        if not bc_digits:
            bc_final = "0"
        else:
            n = int(bc_digits)
            bc_final = str(n) if n in allowed_bc_ids else "0"
    else:
        bc_final = bc_digits if bc_digits else "0"
    sc = key_vals.get("sc_keyword", "") or "원문상 명시 없음"
    tl = key_vals.get("tl_summary", "") or "원문상 명시 없음."
    return (
        f"title: {title}\n"
        f"bc_id: {bc_final}\n"
        f"sc_keyword: {sc}\n"
        f"tl_summary: {tl}"
    )


def _is_empty_like(v: str) -> bool:
    vv = _normalize_ws(v).lower()
    return vv in _EMPTY_TOKENS


def _tl_continuation_line_is_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.startswith("#"):
        return True
    if "세줄 요약" in s.replace(" ", "") and len(s) < 24:
        return True
    if _TL_CONT_NOISE_LINE.match(s):
        return True
    return False


def _sanitize_tl_summary_value(tl: str) -> str:
    tl = _normalize_ws(tl)
    # LLM 출력 접두 노이즈(예: "세줄 요약", "요약 ###") 제거
    while True:
        prev = tl
        tl = re.sub(r"^(세\s*줄\s*요약|세줄요약)\s*[:：\-]?\s*", "", tl, flags=re.IGNORECASE)
        tl = re.sub(r"^요약\s*[:：\-]?\s*", "", tl, flags=re.IGNORECASE)
        tl = re.sub(r"^#{1,6}\s*", "", tl)
        tl = _normalize_ws(tl)
        if tl == prev:
            break
    for prefix in ("국회 의안 통합 요약", "국회 의안 통합 요약관"):
        if tl.startswith(prefix):
            tl = tl[len(prefix) :].strip(" \t:：·-—")
    tl = re.sub(r"#{1,6}\s*[\d\.]*\s*", "", tl)
    tl = re.sub(r"\*{1,2}", "", tl)
    tl = _normalize_ws(tl)
    if not tl:
        return ""
    parts = re.split(r"(?<=[.!?。])\s+", tl)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        parts = [tl]
    out = " ".join(parts[:3])
    out = _normalize_ws(out)
    if len(out) > 520:
        out = out[:520].rsplit(" ", 1)[0]
    if out and not out.endswith((".", "!", "?", "。")):
        out += "."
    return out


def approx_token_count(text: str) -> tuple[int, str]:
    if not text:
        return 0, "empty"
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text)), "tiktoken_cl100k_base"
    except Exception:
        return max(1, len(text) // 2), "chars_div_2"


def sanitize_partial_memo(memo_text: str, seen_fields: dict[str, str]) -> str:
    kept: list[str] = []
    for raw in (memo_text or "").splitlines():
        line = raw.strip()
        if not line:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        norm = _normalize_ws(line)
        if _PROMPT_LEAK_LINE_RE.match(norm):
            continue
        if norm.startswith("아래는 구간에서 추출한 주요 사실을 bullet(-) 목록으로 정리한 내용입니다"):
            continue
        if norm.startswith("[참고: 각 구간별 메모를 종합하여 작성한 요약입니다.]"):
            continue
        m = _PARTIAL_LINE_RE.match(line)
        if not m:
            kept.append(line)
            continue
        key = _normalize_ws(m.group(1))
        val = _normalize_ws(m.group(2))
        if _PROMPT_LEAK_LINE_RE.match(key) or _PROMPT_LEAK_LINE_RE.match(val):
            continue
        if _is_empty_like(val):
            continue
        prev = seen_fields.get(key)
        if prev is not None and prev == val:
            continue
        seen_fields[key] = val
        kept.append(f"- {key}: {val}")
    out = "\n".join(kept).strip()
    return out or "- (이 구간에서 확정 가능한 신규 사실 없음)"


def sanitize_summary_output_text(text: str) -> str:
    if not text:
        return ""
    kept: list[str] = []
    for raw in text.splitlines():
        norm = _normalize_ws(raw)
        if _PROMPT_LEAK_LINE_RE.match(norm):
            continue
        if norm.startswith("아래는 구간에서 추출한 주요 사실을 bullet(-) 목록으로 정리한 내용입니다"):
            continue
        if norm.startswith("[참고: 각 구간별 메모를 종합하여 작성한 요약입니다.]"):
            continue
        kept.append(raw)
    return "\n".join(kept).strip()


def _classify_memo_line(line: str) -> str:
    s = _normalize_ws(line)
    if any(k in s for k in ("제목", "의안번호", "발의자", "발의일", "발의연월일")):
        return "식별정보"
    if any(k in s for k in ("제안이유", "목적", "배경", "쟁점", "문제", "취지")):
        return "핵심쟁점"
    if any(k in s for k in ("개정", "조문", "신구", "권한", "절차", "범위", "제도")):
        return "조문변경"
    if any(k in s for k in ("부칙", "시행", "경과", "적용", "집행")):
        return "시행·부칙"
    return "기타"


def build_structured_memo_text(memos: list[str]) -> str:
    sec_lines: dict[str, list[str]] = {k: [] for k in _FINAL_SECTION_ORDER}
    seen: set[str] = set()
    for memo in memos:
        for raw in (memo or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            if not line.startswith("-"):
                line = f"- {line}"
            key = _normalize_ws(line)
            if key in seen:
                continue
            seen.add(key)
            sec = _classify_memo_line(line)
            sec_lines[sec].append(line)
    blocks: list[str] = []
    for sec in _FINAL_SECTION_ORDER:
        lines = sec_lines[sec]
        if not lines:
            continue
        blocks.append(f"[{sec}]\n" + "\n".join(lines))
    return "\n\n".join(blocks).strip()


def trim_structured_memo_text_by_budget(structured_text: str, budget_tokens: int) -> tuple[str, bool, int]:
    curr_tokens, _ = approx_token_count(structured_text)
    if curr_tokens <= budget_tokens:
        return structured_text, False, curr_tokens

    sec_map: dict[str, list[str]] = {}
    current: str | None = None
    for raw in structured_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = line[1:-1].strip()
            if sec in _FINAL_SECTION_ORDER:
                current = sec
                sec_map.setdefault(sec, [])
            else:
                current = None
            continue
        if current is not None:
            sec_map.setdefault(current, []).append(line)

    drop_order = ["기타", "시행·부칙", "조문변경", "핵심쟁점", "식별정보"]
    for sec in drop_order:
        lines = sec_map.get(sec, [])
        while lines:
            lines.pop()
            blocks: list[str] = []
            for k in _FINAL_SECTION_ORDER:
                ls = sec_map.get(k, [])
                if ls:
                    blocks.append(f"[{k}]\n" + "\n".join(ls))
            cand = "\n\n".join(blocks).strip()
            curr_tokens, _ = approx_token_count(cand)
            if curr_tokens <= budget_tokens:
                return cand, True, curr_tokens
        sec_map[sec] = lines

    fallback = structured_text.split("\n")
    out_lines: list[str] = []
    for ln in fallback:
        out_lines.append(ln)
        cand = "\n".join(out_lines).strip()
        t, _ = approx_token_count(cand)
        if t > budget_tokens:
            out_lines.pop()
            break
    final_text = "\n".join(out_lines).strip()
    final_tokens, _ = approx_token_count(final_text)
    return final_text, True, final_tokens


def _extract_sc_keyword_candidates(text_blob: str, limit: int = 2) -> list[str]:
    text = _normalize_ws(text_blob)
    if not text:
        return []
    tokens = re.findall(r"[가-힣A-Za-z0-9·]{2,}", text)
    freq: dict[str, int] = {}
    for t in tokens:
        t = t.strip("·")
        if len(t) < 2:
            continue
        if t in _SC_KEYWORD_STOPWORDS:
            continue
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))
    return [k for k, _ in ranked[:limit]]


def _looks_generic_title(title: str) -> bool:
    t = _normalize_ws(title)
    if not t or t == "원문상 명시 없음":
        return True
    if len(t) <= 6:
        return True
    for prefix in _GENERIC_TITLE_PREFIXES:
        if t.startswith(prefix):
            return True
    return False


def _extract_title_candidate_from_summary(summary_text: str) -> str:
    lines = [ln.strip() for ln in (summary_text or "").splitlines() if ln.strip()]
    for ln in lines:
        if ln.startswith("#"):
            continue
        if re.match(r"^\d+\.", ln):
            continue
        if ln in {"문제의 실체", "제도 설계", "이해관계 쟁점", "시행/집행 영향", "결론", "추가 고려사항"}:
            continue
        first = re.split(r"[.!?。]", ln, maxsplit=1)[0].strip()
        if not first:
            continue
        if len(first) > 70:
            first = first[:70].rsplit(" ", 1)[0]
        m = re.search(r"([가-힣A-Za-z0-9·\-\s]{6,}(?:법안|개정법률안|제정안))", first)
        if m:
            return _normalize_ws(m.group(1))
        return _normalize_ws(first)
    return ""


def repair_sc_keyword_from_summary(
    normalized_meta: str,
    summary_text: str,
    *,
    source_full_text: str | None = None,
) -> str:
    lines = normalized_meta.splitlines()
    key_vals: dict[str, str] = {}
    for ln in lines:
        if ":" in ln:
            k, v = ln.split(":", 1)
            key_vals[k.strip()] = v.strip()

    raw_title = strip_metadata_title_suffixes(key_vals.get("title", ""))
    if source_full_text and str(source_full_text).strip() and (
        _looks_generic_title(raw_title) or not raw_title.strip()
    ):
        key_vals["title"] = reconcile_title_for_metadata(
            raw_title,
            source_text=source_full_text,
            summary_text=summary_text,
        )
    elif _looks_generic_title(raw_title):
        cand_title = _extract_title_candidate_from_summary(summary_text)
        if cand_title:
            key_vals["title"] = cand_title
        elif not raw_title:
            key_vals["title"] = "원문상 명시 없음"
    else:
        key_vals["title"] = raw_title or "원문상 명시 없음"

    title_final = key_vals.get("title", "원문상 명시 없음")
    blob = candidate_blob_for_keywords(
        source_text=source_full_text,
        summary_text=summary_text,
        title=title_final,
    )
    blob_lower = blob.lower()

    raw_sc = scrub_sc_keyword_phrase(key_vals.get("sc_keyword", ""))
    parsed = [x.strip() for x in raw_sc.split(",") if x.strip()] if raw_sc else []
    valid = [
        k
        for k in parsed
        if k and k.lower() in blob_lower and k not in _SC_KEYWORD_STOPWORDS
    ]
    if valid and not sc_keyword_parts_plausible(valid, blob):
        valid = []

    if not valid:
        valid = _extract_sc_keyword_candidates(blob, limit=2)
    if not valid:
        valid = extract_keyword_candidates_from_title(title_final)
    if valid:
        key_vals["sc_keyword"] = ", ".join(valid[:2])
    else:
        key_vals["sc_keyword"] = "원문상 명시 없음"

    raw_tl = key_vals.get("tl_summary", "") or ""
    if source_full_text and str(source_full_text).strip() and tl_summary_needs_source_fallback(
        raw_tl
    ):
        fb = fallback_tl_summary_from_source(source_full_text)
        if fb:
            key_vals["tl_summary"] = _sanitize_tl_summary_value(fb) or fb

    tl_out = key_vals.get("tl_summary", "원문상 명시 없음.")
    tl_out = _sanitize_tl_summary_value(tl_out) or tl_out
    if not tl_out:
        tl_out = "원문상 명시 없음."
    key_vals["tl_summary"] = tl_out

    return (
        f"title: {key_vals.get('title', '원문상 명시 없음')}\n"
        f"bc_id: {key_vals.get('bc_id', '0')}\n"
        f"sc_keyword: {key_vals.get('sc_keyword', '원문상 명시 없음')}\n"
        f"tl_summary: {key_vals['tl_summary']}"
    )


def normalize_metadata_text(raw: str) -> str:
    vals: dict[str, str] = {"title": "", "bc_id": "0", "sc_keyword": "", "tl_summary": ""}
    current_key: str | None = None
    for raw_line in (raw or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched = False
        for key, pat in _META_KEY_PATTERNS.items():
            m = pat.match(line)
            if m:
                vals[key] = _normalize_ws(m.group(1))
                current_key = key
                matched = True
                break
        if matched:
            continue
        if current_key:
            if current_key == "tl_summary" and _tl_continuation_line_is_noise(line):
                continue
            vals[current_key] = _normalize_ws((vals[current_key] + " " + line).strip())

    vals["title"] = strip_metadata_title_suffixes(vals["title"])
    if not vals["title"]:
        vals["title"] = "원문상 명시 없음"
    bc_digits = re.sub(r"[^\d]", "", vals["bc_id"])
    vals["bc_id"] = bc_digits if bc_digits else "0"
    if not vals["sc_keyword"]:
        vals["sc_keyword"] = "원문상 명시 없음"

    tl = vals["tl_summary"]
    tl = re.sub(r"\s*[①②③④⑤]\s*", " ", tl)
    tl = re.sub(r"\s+", " ", tl).strip()
    tl = _sanitize_tl_summary_value(tl)
    if not tl:
        tl = "원문상 명시 없음."
    vals["tl_summary"] = tl

    return (
        f"title: {vals['title']}\n"
        f"bc_id: {vals['bc_id']}\n"
        f"sc_keyword: {vals['sc_keyword']}\n"
        f"tl_summary: {vals['tl_summary']}"
    )
