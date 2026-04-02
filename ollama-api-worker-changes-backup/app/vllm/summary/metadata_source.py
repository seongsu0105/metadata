# =============================================================================
#   app\vllm\summary\metadata_source.py
# =============================================================================
"""
원문(OCR 합본 등) 기준 메타데이터 보강 처리.

데이터셋 파이프라인·Worker `repair_sc_keyword_from_summary`에서 공용으로 쓸 수 있다.
"""

from __future__ import annotations

import re


_TITLE_HEAD_CHARS = 300
_TITLE_TAIL_FALLBACK = 80


def _normalize_title_ws(title: str) -> str:
    t = (title or "").replace("  ", " ").strip()
    return re.sub(r"\s+", " ", t)


def extract_title_from_source(text: str) -> str:
    """원문 앞부분에서 법률안 제목 후보를 휴리스틱으로 추출."""
    if not text:
        return ""
    head = text[:_TITLE_HEAD_CHARS]
    m = re.search(r"([가-힣\s]+법률안)", head)
    if m:
        return _normalize_title_ws(m.group(1))
    return _normalize_title_ws(text[:_TITLE_TAIL_FALLBACK])


def reconcile_title_for_metadata(
    model_title: str,
    *,
    source_text: str | None,
    summary_text: str,
    title_min_guess_chars: int = 8,
    title_presence_check_chars: int = 5000,
) -> str:
    """
    모델 title을 원문·요약 맥락에서 다듬는다.
    `source_text`가 없으면 `summary_text`만으로 `extract_title_from_source`에 넘길 수 없으므로
    model_title를 정규화해 반환한다.
    """
    base = (model_title or "").strip()
    if not source_text or not str(source_text).strip():
        return _normalize_title_ws(base)
    fb = extract_title_from_source(source_text)
    if not base:
        return _normalize_title_ws(fb)
    base_n = _normalize_title_ws(base)
    fb_n = _normalize_title_ws(fb)
    win = source_text[:title_presence_check_chars]
    if len(base_n) < title_min_guess_chars and "법" not in base_n:
        return fb_n or base_n
    if (
        fb_n
        and base_n not in win
        and fb_n in win
        and len(fb_n) + 1 >= len(base_n)
    ):
        return fb_n
    return base_n


# --- sc_keyword ---

_KEYWORD_SUFFIXES = (
    "일부개정법률안",
    "전부개정법률안",
    "에관한특별법",
    "을위한특별법",
    "에관한법률",
    "을위한법률",
    "특별법",
    "법률안",
    "법률",
)


def scrub_sc_keyword_phrase(keyword: str, *, latin_strip_min: int = 5) -> str:
    """허용 문자만 남기고 최대 2토큰(쉼표 구분)으로 정리."""
    parts = re.split(r"[,，]", keyword or "")
    cleaned: list[str] = []
    for p in parts:
        t = re.sub(r"[^\s0-9가-힣·]", "", p).strip()
        t = re.sub(rf"\s*[A-Za-z]{{{latin_strip_min},}}\s*", " ", t)
        t = re.sub(r"\s+", "", t)
        if t:
            cleaned.append(t)
    return ", ".join(cleaned[:2]) if cleaned else ""


def sc_keyword_parts_plausible(parts: list[str], blob: str) -> bool:
    if not parts:
        return False
    blob_c = re.sub(r"\s+", "", blob or "")
    usable = [p.strip() for p in parts if len(p.strip()) >= 2]
    if not usable:
        return False
    hits = sum(
        1 for p in usable if p in (blob or "") or re.sub(r"\s+", "", p) in blob_c
    )
    return hits >= (len(usable) + 1) // 2


def extract_keyword_candidates_from_title(title: str) -> list[str]:
    """
    제목에서 키워드 후보 최대 2개(Kiwi 없이 접미 제거·명사구 분할).
    """
    t = _normalize_title_ws(title)
    if not t:
        return []
    clean = t
    for s in _KEYWORD_SUFFIXES:
        clean = clean.replace(s, "")
    clean = clean.strip()
    refined: list[str] = []
    for p in clean.split():
        r = re.sub(
            r"(을|를|이|가|은|는|의|에|로|와|과|및|등|관한|안|·)+$",
            "",
            p,
        ).strip()
        if len(r) >= 2:
            refined.append(r)
    if len(refined) >= 2:
        return refined[
            :2
        ]
    word = refined[0] if refined else clean
    if not word:
        return []
    if len(word) <= 3:
        return [word]
    half = len(word) // 2
    a, b = word[:half].strip(), word[half:].strip()
    return [x for x in (a, b) if x][:2]


def candidate_blob_for_keywords(
    *,
    source_text: str | None,
    summary_text: str,
    title: str,
    max_source_chars: int = 12000,
) -> str:
    head = (source_text or "")[:max_source_chars] if source_text else ""
    return f"{head}\n{summary_text or ''}\n{title or ''}"


# --- tl_summary 원문 기반 보조(세 문장 미달 등) ---

_TL_MIN_PERIODS = 2
_TL_MIN_CHARS = 40

_SUMMARY_CONTAMINATION = re.compile(
    r"\(생략\)|1\s*\.?\s*∼\s*\d+|제\s*\d+\s*조\s*\(\s*정의\s*\)|-{10,}|<신설\s*>",
    re.IGNORECASE,
)

_TABLE_MARKER = re.compile(
    r"의안\s*명|대표\s*발의|발의\s*일자|심사\s*경과|법률제\s*호|신[ㆍ·]\s*구조문|건명",
    re.IGNORECASE,
)


def tl_summary_needs_source_fallback(tl: str) -> bool:
    s = (tl or "").strip()
    if len(s) < _TL_MIN_CHARS:
        return True
    if _SUMMARY_CONTAMINATION.search(s):
        return True
    if _TABLE_MARKER.search(s):
        return True
    if s.endswith("건명"):
        return True
    periods = s.count(".") + s.count("。")
    if periods < _TL_MIN_PERIODS:
        return True
    if len(s) > 80 and not re.search(r"[.!?。…]['\"」\s]*$", s):
        return True
    return False


def fallback_tl_summary_from_source(source: str) -> str | None:
    """
    제안이유 구간 등에서 짧은 세 문장형 tl_summary 후보를 규칙으로 만든다.
    실패 시 None.
    """
    if not source or len(source.strip()) < 80:
        return None
    text = re.sub(r"-\s*\d+\s*-", " ", source)
    text = re.sub(r"\s+", " ", text)
    start_m = re.search(r"(?:2\.\s*)?대안의\s*제안\s*이유|제안\s*이유", text)
    if start_m:
        target = text[start_m.end() :]
    else:
        parts = re.split(
            r"(?:제안\s*이유|제안이유)(?:및주요내용)?|주요\s*내용",
            text,
            maxsplit=1,
        )
        target = parts[-1] if len(parts) > 1 else text[:1000]

    end_m = re.search(
        r"(?:3\.\s*)?대안의\s*주요\s*내용|법률제\s*호|부\s*칙|신[ㆍ·]\s*구조문|의안\s*명|심사\s*경과",
        target,
    )
    if end_m:
        target = target[: end_m.start()]

    target = re.sub(r"\(안제[^)]+\)", "", target)
    target = re.sub(r"의안\s*번호.*$", "", target, flags=re.DOTALL)
    target = re.sub(r"\s+", " ", target).strip()

    cands = re.split(r"(?<=[\.\?!다음임함됨])\s+", target)
    sentences: list[str] = []
    for c in cands:
        s = c.strip(" .")
        if len(s) < 20:
            continue
        if _TABLE_MARKER.search(s):
            continue
        if s.startswith(("가.", "나.", "다.", "라.", "마.")):
            continue
        sentences.append(s + ".")

    if not sentences:
        fallback = target[:420].strip() or str(source)[:420].strip()
        if not fallback:
            return None
        out = fallback if fallback.endswith(".") else (fallback + ".")
    else:
        out = " ".join(sentences[:3]).strip()
        if not out.endswith("."):
            out += "."
    if tl_summary_needs_source_fallback(out):
        return None
    return out
