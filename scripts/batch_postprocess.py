from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass

from kiwipiepy import Kiwi

from batch_config import SANITIZE_OVERRIDES_JSON

# ---------------------------------------------------------------------------
# 메타데이터 후처리 규칙 (문서 종류와 무관하게 동일 적용)
# R1  유니코드 NFC
# R2  한글 음절–종성(U+11A8–11FF) 사이 공백 제거 후 재결합
# R3  조사 앞 불필요 공백(해임 을 등) 정리
# R4  법령 수치에서 흔한 숫자 분절(1 00분의 등) 공백 제거
# R5  sc_keyword·요약: 라틴 알파벳 연속 토큰(5자 이상) 제거(외국어 혼입 완화)
# R6  요약: 조문·생략·정의 붙여넣기 패턴이면 규칙 추출로 교체
# R7  요약: 라틴/이종 문자 비율·베트남어 문자 존재 시 규칙 추출
# R8  제목: 원문 부재·과도하게 짧으면 휴리스틱 제목
# R9  키워드: 본문 출현 과반 미달 시 제목 기반 재추출 + 허용 문자 집합
# R10 요약: 문장 부호·길이 기반 비정상 종료 시 규칙 추출
# R11 선택: metadata_sanitize_overrides.json 의 from→to (조직별만)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetadataPostprocessRules:
    summary_min_chars: int = 40
    summary_min_sentence_punct: int = 2
    summary_max_latin_to_hangul_ratio: float = 0.22
    summary_latin_count_for_ratio: int = 18
    keyword_plausibility_blob_chars: int = 12000
    title_presence_check_chars: int = 5000
    title_min_guess_chars: int = 8
    latin_token_strip_min_len: int = 5


RULES = MetadataPostprocessRules()
_OVERRIDES_CACHE: list[tuple[str, str]] | None = None
kiwi = Kiwi()


def _load_sanitize_overrides() -> list[tuple[str, str]]:
    global _OVERRIDES_CACHE
    if _OVERRIDES_CACHE is not None:
        return _OVERRIDES_CACHE
    acc: list[tuple[str, str]] = []
    if os.path.isfile(SANITIZE_OVERRIDES_JSON):
        try:
            with open(SANITIZE_OVERRIDES_JSON, encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                if isinstance(item, dict) and "from" in item and "to" in item:
                    a, b = str(item["from"]), str(item["to"])
                    if a:
                        acc.append((a, b))
        except (OSError, json.JSONDecodeError):
            pass
    _OVERRIDES_CACHE = acc
    return acc


def normalize_hangul_syllables(s: str) -> str:
    """
    GPU와 무관. 모델/Kiwi 출력에서 음절과 종성 자모가 공백으로 떨어지는 경우(하 ᆯ)를
    붙인 뒤 NFC로 완성형으로 만든다. (U+11A8–U+11FF: Hangul Jongseong)
    """
    if not s:
        return s
    t = s
    for _ in range(4):
        t2 = re.sub(r"([가-힣])\s+([\u11A8-\u11FF])", r"\1\2", t)
        t2 = unicodedata.normalize("NFC", t2)
        if t2 == t:
            break
        t = t2
    t = re.sub(r"([가-힣])\s+([음함임됨])(?=[\s\.,]|$)", r"\1\2", t)
    return unicodedata.normalize("NFC", t)


def _apply_numeric_spacing_rule(s: str) -> str:
    """R4: 법령 숫자 표기에서 한 자리·다자리 사이 잘못 든 공백 제거."""
    t = re.sub(r"(\d)\s+(\d{2,})\s*(?=[분년월일%]|$)", r"\1\2", s)
    return re.sub(r"(\d)\s+(\d{2,})(?=\s*[가-힣])", r"\1\2", t)


def _strip_long_latin_tokens(s: str) -> str:
    """R5: 5자 이상 연속 라틴 알파벳 토큰을 공백으로 제거(약어 남기지 않음)."""
    mnl = RULES.latin_token_strip_min_len
    t = re.sub(rf"\s*[A-Za-z]{{{mnl},}}\s*", " ", s)
    return re.sub(r"\s+", " ", t).strip()


_SUMMARY_CONTAMINATION_RE = re.compile(
    r"\(생략\)|1\s*\.?\s*∼\s*\d+|제\s*\d+\s*조\s*\(\s*정의\s*\)|-{10,}|<신설\s*>",
    re.IGNORECASE,
)

_VIETNAM_LATIN_RE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩị"
    r"òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]"
)

_SUMMARY_NOISE_RE = re.compile(
    r"[ぁ-ゖァ-ヺ々〆〤]|ことを|进行|same|etc\.|자sal|放送|공공机构",
    re.IGNORECASE,
)

_SUMMARY_TABLE_MARKER_RE = re.compile(
    r"의안\s*명|대표\s*발의|발의\s*일자|심사\s*경과|법률제\s*호|신[ㆍ·]\s*구조문|건명",
    re.IGNORECASE,
)


def summary_should_use_extract_fallback(summary: str) -> bool:
    """R6·R7: 조문 베끼기·이종어 비율."""
    s = summary or ""
    if _SUMMARY_CONTAMINATION_RE.search(s):
        return True
    if _SUMMARY_TABLE_MARKER_RE.search(s):
        return True
    if _VIETNAM_LATIN_RE.search(s):
        return True
    h = len(re.findall(r"[가-힣]", s))
    lat = len(re.findall(r"[A-Za-z]", s))
    if h == 0 and lat > 8:
        return True
    if lat >= RULES.summary_latin_count_for_ratio and h > 0:
        if (lat / h) > RULES.summary_max_latin_to_hangul_ratio:
            return True
    return False


def summary_quality_score(summary: str) -> int:
    """요약 텍스트의 품질 점수(0~5). 점수가 높을수록 채택 우선."""
    s = (summary or "").strip()
    if not s:
        return 0
    score = 0
    if len(s) >= 80:
        score += 1
    periods = s.count(".") + s.count("。")
    if periods >= RULES.summary_min_sentence_punct:
        score += 1
    if not _SUMMARY_NOISE_RE.search(s):
        score += 1
    hangul = len(re.findall(r"[가-힣]", s))
    latin = len(re.findall(r"[A-Za-z]", s))
    if hangul >= 50:
        score += 1
    if latin <= 8 or (hangul > 0 and (latin / max(hangul, 1)) <= 0.12):
        score += 1
    return score


def is_tl_summary_truncated_or_broken(s: str) -> bool:
    """R10: 특정 단어 나열 없이 길이·마침표 개수·끝 문장부호로만 판단."""
    t = (s or "").strip()
    if len(t) < RULES.summary_min_chars:
        return True
    if _SUMMARY_TABLE_MARKER_RE.search(t):
        return True
    if t.endswith("건명"):
        return True
    periods = t.count(".") + t.count("。")
    if periods < RULES.summary_min_sentence_punct:
        return True
    if len(t) > 80 and not re.search(r"[.!?。…]['\"」\s]*$", t):
        return True
    return False


def summary_is_acceptable(summary: str) -> bool:
    """저장 가능한 최소 품질 기준."""
    s = (summary or "").strip()
    if not s:
        return False
    if _SUMMARY_NOISE_RE.search(s):
        return False
    if is_tl_summary_truncated_or_broken(s):
        return False
    return summary_quality_score(s) >= 4


def scrub_sc_keyword(keyword: str) -> str:
    """R9: 허용 문자 집합(한글·숫자·쉼표·가운뎃점·공백)."""
    parts = re.split(r"[,，]", keyword)
    cleaned: list[str] = []
    for p in parts:
        t = re.sub(r"[^\s0-9가-힣·]", "", p).strip()
        t = _strip_long_latin_tokens(t)
        t = re.sub(r"\s+", "", t)
        if t:
            cleaned.append(t)
    return ", ".join(cleaned[:2]) if cleaned else ""


def sc_keyword_plausible(keyword: str, text: str, title: str) -> bool:
    if not keyword or len(keyword.strip()) < 2:
        return False
    blob = (text[: RULES.keyword_plausibility_blob_chars] if text else "") + (
        title or ""
    )
    blob_c = re.sub(r"\s+", "", blob)
    parts = [p.strip() for p in re.split(r"[,，]", keyword) if len(p.strip()) >= 2]
    if not parts:
        return False
    hits = sum(1 for p in parts if p in blob or re.sub(r"\s+", "", p) in blob_c)
    return hits >= (len(parts) + 1) // 2


def normalize_title(title: str) -> str:
    title = title.replace("  ", " ").strip()
    title = re.sub(r"\s+", " ", title)
    return title


def extract_title(text: str) -> str:
    match = re.search(r"([가-힣\s]+법률안)", text[:300])
    if match:
        return normalize_title(match.group(1))
    return normalize_title(text[:80])


def normalize_llm_output_text(s: str) -> str:
    """R1–R4·R11 + 공통 공백 정리."""
    if not s:
        return s
    t = s.replace("\ufeff", "")
    for a, b in sorted(_load_sanitize_overrides(), key=lambda x: -len(x[0])):
        t = t.replace(a, b)
    t = _apply_numeric_spacing_rule(t)
    t = re.sub(
        r"([가-힣]{2,})\s+(을|를|이|가|은|는|과|와)(?=[가-힣\s]|$)",
        r"\1\2",
        t,
    )
    t = normalize_hangul_syllables(t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return unicodedata.normalize("NFC", t)


def reconcile_title_with_text(model_title: str, text: str) -> str:
    """R8."""
    fb = extract_title(text)
    base = (model_title or "").strip()
    if not base:
        return normalize_llm_output_text(fb)
    base_n = normalize_llm_output_text(base)
    fb_n = normalize_llm_output_text(fb)
    if len(base_n) < RULES.title_min_guess_chars and "법" not in base_n:
        return fb_n
    if (
        base_n not in text[: RULES.title_presence_check_chars]
        and fb_n in text[: RULES.title_presence_check_chars]
        and len(fb_n) + 1 >= len(base_n)
    ):
        return fb_n
    return base_n


def loosen_dense_hangul_summary(s: str) -> str:
    """공백이 거의 없는 '벽돌' 한글 요약을 형태소 경계로 띄어쓴다."""
    if not s or len(s) < 35:
        return s
    if (s.count(" ") / len(s)) > 0.045:
        return s
    try:
        out = " ".join(tok.form for tok in kiwi.tokenize(s))
        return normalize_hangul_syllables(out)
    except Exception:
        return s


def extract_keyword(title: str) -> str:
    suffixes = [
        "일부개정법률안",
        "전부개정법률안",
        "에관한특별법",
        "을위한특별법",
        "에관한법률",
        "을위한법률",
        "특별법",
        "법률안",
        "법률",
    ]
    clean_title = title
    for s in suffixes:
        clean_title = clean_title.replace(s, "")
    clean_title = clean_title.strip()

    parts = clean_title.split()
    refined_parts = []
    for p in parts:
        refined = re.sub(
            r"(을|를|이|가|은|는|의|에|로|와|과|및|등|관한|안|·)+$", "", p
        ).strip()
        if len(refined) >= 2:
            refined_parts.append(refined)

    if len(refined_parts) >= 2:
        return ", ".join(refined_parts[:2])

    word = refined_parts[0] if refined_parts else clean_title
    tokens = kiwi.tokenize(word)
    nouns = [
        token.form
        for token in tokens
        if token.tag in ("NNG", "NNP") and len(token.form) >= 2
    ]

    if len(nouns) >= 2:
        return f"{nouns[0]}, {nouns[1]}"
    if len(nouns) == 1:
        return nouns[0]

    if len(word) <= 3:
        return word
    half = len(word) // 2
    return f"{word[:half]}, {word[half:]}"


def extract_summary(text: str) -> str:
    text = re.sub(r"-\s*\d+\s*-", " ", text)
    text = re.sub(r"\s+", " ", text)
    start_m = re.search(r"(?:2\.\s*)?대안의\s*제안\s*이유|제안\s*이유", text)
    if start_m:
        target = text[start_m.end() :]
    else:
        parts = re.split(
            r"(?:제안\s*이유|제안이유)(?:및주요내용)?|주요\s*내용", text, maxsplit=1
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
        if _SUMMARY_TABLE_MARKER_RE.search(s):
            continue
        if s.startswith(("가.", "나.", "다.", "라.", "마.")):
            continue
        sentences.append(s + ".")

    if not sentences:
        fallback = target[:420].strip() or text[:420].strip()
        return fallback if fallback.endswith(".") else (fallback + ".")

    out = " ".join(sentences[:3]).strip()
    if not out.endswith("."):
        out += "."
    return out


def format_metadata_block(
    title: str, bc_id: int, sc_keyword: str, tl_summary: str
) -> str:
    return (
        f"title: {title}\n"
        f"bc_id: {bc_id}\n"
        f"sc_keyword: {sc_keyword}\n"
        f"tl_summary: {tl_summary}"
    )
