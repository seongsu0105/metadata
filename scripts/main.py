import os
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any
import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ReadTimeout, Timeout
from PyPDF2 import PdfReader
from kiwipiepy import Kiwi

# ---------------------------------------------------------------------------
# 산출물
# - metadata.jsonl: DB·배포용 순수 메타데이터 (본 역할)
# - finetune_dataset.jsonl: 메타데이터(key:값 4줄) LoRA 학습용 (선택)
#   → Worker의 metadata_user_prompt + METADATA_SYSTEM 과 동일 계약이어야
#     추론·학습이 맞물린다. (docs/ex/summary_prompts.py METADATA_* 와 동일)
# - 6단원 마크다운 요약(SUMMARY_SYSTEM_FINAL 등)은 태스크가 다름.
#   본 스크립트는 생성하지 않음. 필요 시 Worker·별도 파이프라인으로
#   instruction/output 형식이 다른 JSONL을 만든다.
# ---------------------------------------------------------------------------
METADATA_JSONL = "metadata2.jsonl"
FINETUNE_JSONL = "finetune_dataset2.jsonl"
WRITE_FINETUNE = True
# 학습 input 상한·앵커(규칙 R10)
FINETUNE_INPUT_MAX_CHARS = 6000
FINETUNE_ANCHOR_MARGIN = 400

PDF_DIR = "pdfdata2"
# 동일 폴더에 두면 from→to 치환을 추가 적용 (도메인 전용 보정, 없어도 동작)
SANITIZE_OVERRIDES_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "metadata_sanitize_overrides.json"
)
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
# 메타 4줄만 생성 시 1024면 대부분 충분. 잘림이 잦으면 1280~1536으로 올리기.
OLLAMA_NUM_PREDICT = 1024
# Ollama가 응답을 끝낼 때까지 기다리는 최대 시간(초).
# 짧으면 Read timed out — PDF가 길거나 GPU가 느리면 300초를 넘기기 쉽다.
OLLAMA_CONNECT_TIMEOUT = 30
OLLAMA_READ_TIMEOUT = 900
# 타임아웃·일시 연결 실패 시 재시도 횟수(추가)
OLLAMA_HTTP_RETRIES = 2
METADATA_QUALITY_RETRIES = 2

# ----- Worker summary_prompts.py 와 동일 계약 (메타데이터 전용) -----
BASE_SYSTEM_COMMON = (
    "공통 규칙:\n"
    "- 제공된 텍스트 근거만 사용하고 추정/환각 금지\n"
    "- 원문에 없는 수치·날짜·고유명사를 지어내지 말 것\n"
    "- 불명확한 항목은 단정하지 말고 '원문상 명시 없음'으로 처리\n"
    "- 불필요한 인사말/변명/자기설명 금지\n"
)

METADATA_SYSTEM = (
    BASE_SYSTEM_COMMON
    + "역할: [대상 텍스트]만 보고 DB 저장용 메타데이터를 키:값 한 줄씩 출력한다.\n"
    + "출력은 검색 인덱싱 품질 최적화를 목표로 한다.\n\n"
    + "[절대 순서 — 반드시 이 순서로 4줄만 출력]\n"
    + "1) 첫 줄: title: (실제 제목 문자열)\n"
    + "2) 둘째 줄: bc_id: (숫자)\n"
    + "3) 셋째 줄: sc_keyword: (핵심 단어 최대 2개, 반드시 쉼표로만 구분. 예: 전력, 송전)\n"
    + "4) 넷째 줄: tl_summary: (세 문장, 한 줄로)\n\n"
    + "[title — 최우선·생략 절대 금지]\n"
    + "- 출력의 반드시 첫 번째 줄은 title: 로 시작해야 한다.\n"
    + '- title 값은 빈 칸·"제목 없음"·"N/A"·"해당 없음"·"미정" 금지. 반드시 구체적인 법률안·안건 명칭을 한글로 채운다.\n'
    + "- 찾는 순서: (가) '## 제목' 또는 '1. ## 제목' 바로 아래 한 줄 (나) '의안명'·'법률안 명' 근처 문장 (다) 첫머리에 나온 안건명·법령명 전체 (라) 그래도 없으면 문서 맨 앞 80자 이내 핵심 명사구를 붙여 하나의 제목 문장으로 만든다(빈 값 금지).\n"
    + "- 요약에 '## 제목 (○○법률안)'처럼 나오면 title 값은 괄호 안 ○○만 넣는다. '## 제목', '#', '제목' 레이블·괄호 자체는 title에 넣지 않는다.\n"
    + "- 마크다운 기호(##, **)는 title 값에 넣지 말고 법률안·안건 공식 명칭 문장만 한 줄로.\n\n"
    + "[bc_id]\n"
    + "- 사용자 메시지에 붙은 [빅카테고리(bc_id) 선택 목록]에 있는 id 숫자만. "
    + "목록에 없는 숫자·임의 숫자 금지. 애매하면 목록 중 가장 가까운 하나.\n\n"
    + "[sc_keyword]\n"
    + "- 법안이 직접 다루는 대상(직군·계층·제도명 등) 단어 최대 2개.\n"
    + "- 지나치게 일반적인 단어(정책, 법률, 개선, 제도)는 지양.\n"
    + "- '조사(~및, ~에 관한, ~등)'를 제거한 완성된 명사형으로 출력. 예: '공예문화산업진흥법및' (X) → '공예문화산업법' (O)\n\n"
    + "[tl_summary]\n"
    + "- 반드시 세 문장(마침표 3개 이상). ①개정 결론(무엇이 어떻게 바뀌는지) ②변경 전·후 ③시행일. "
    + '"요약 없음"·한 문장만·빈 값 금지.\n'
    + "- 완성형 한글 음절로만 작성. 음절과 받침(종성) 사이에 공백 금지.\n"
    + "- 조문 통째 인용·'(생략)'·정의 조항만 베끼지 말고 제안 취지 수준으로 요약.\n\n"
    + "[공통]\n"
    + "- 설명·인사·JSON·코드펜스·추가 줄 금지. 위 4키만, 키 이름 철자 정확히 title, bc_id, sc_keyword, tl_summary.\n"
)

# ----- Worker: 6단원 요약·구간 추출 (학습 데이터는 별도 JSONL·별도 생성 파이프라인) -----
SUMMARY_SYSTEM_FINAL = (
    BASE_SYSTEM_COMMON
    + "역할: 국회 의안 분석 요약관.\n"
    + "법률 원문/구간메모를 근거로 요약문만 출력하세요. 마크다운 형식.\n"
    + "6단원 반드시 순서대로, 모두 작성. 생략 금지.\n"
    + "분량은 700~1100자 권장. 지나치게 짧은 제목형 출력 금지.\n"
    + "1. ## 제목 \n"
    + "2. ## 의안번호·발의연원일·발의자\n"
    + "3. ## 제안이유 및 주요내용\n"
    + "4. ## 개정법률안\n"
    + "5. ## 부칙\n"
    + "6. ## 신·구조문대비표\n"
    + "핵심만 간결히 쓰되, 사안의 배경·쟁점·변경 전후를 빠뜨리지 마세요. 조문 통째 인용 금지.\n"
)

SUMMARY_SYSTEM_PARTIAL = (
    BASE_SYSTEM_COMMON
    + "역할: 장문 법안을 구간 단위로 추출하는 팩트 추출기.\n"
    + "구간만 읽고 사실만 bullet(-)로 짧게 작성.\n"
    + "JSON·인사말·## 대제목·6단원 구조 금지. 반복 금지.\n"
)

# ==============================
# 1. 카테고리
# ==============================

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

# ==============================
# 2. bc_id
# ==============================


def assign_bc_id(text: str) -> int:
    category_map = {
        1: ["정치", "행정", "지방자치", "공무원", "선거", "자치단체"],
        2: [
            "경제",
            "산업",
            "금융",
            "조세",
            "세법",
            "기업",
            "공정거래",
            "재정",
            "조세특례",
            "상속세",
            "증여세",
            "자본시장",
            "펀드",
        ],
        3: ["복지", "장애인", "보건", "의료", "노동", "가족", "연금", "유족"],
        4: [
            "법사",
            "사법",
            "형법",
            "민법",
            "범죄",
            "안전",
            "형사소송",
            "민사소송",
            "공중협박",
            "처벌",
            "벌금",
            "징역",
            "병역",
        ],
        5: [
            "교육",
            "문화",
            "콘텐츠",
            "예술",
            "체육",
            "관광",
            "언론",
            "방송",
            "박물관",
            "미술관",
            "공예",
            "대중문화",
        ],
        6: [
            "과학",
            "기술",
            "정보통신",
            "인공지능",
            "데이터",
            "원자력",
            "정보보호",
            "정보통신망",
        ],
        7: ["국방", "외교", "통일", "군사", "안보"],
        8: ["환경", "에너지", "기후", "자원", "폐기물"],
        9: ["농림", "축산", "식품", "해양", "수산"],
        10: [
            "국토",
            "교통",
            "주택",
            "건설",
            "도로",
            "철도",
            "항공",
            "소방",
            "하도급",
            "계약",
        ],
    }

    scores = {k: 0 for k in category_map}
    for bc_id, keywords in category_map.items():
        for kw in keywords:
            scores[bc_id] += text.count(kw)
    return max(scores, key=scores.get)


def normalize_bc_id(raw: Any, text: str) -> int:
    """모델이 의안번호 등을 넣으면 1~10이 아니므로 휴리스틱으로 교정."""
    if raw is None:
        return assign_bc_id(text)
    s = str(raw).strip()
    m = re.search(r"(\d+)", s)
    if m:
        n = int(m.group(1))
        if n == 0:
            return assign_bc_id(text)
        if n in VALID_BC_IDS:
            return n
    return assign_bc_id(text)


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


def generate_summary_ollama(text: str, bc_block: str) -> str:
    prompt = metadata_user_prompt(text, bc_block)
    payload = {
        "model": OLLAMA_MODEL,
        "system": METADATA_SYSTEM,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
    }
    timeout = (OLLAMA_CONNECT_TIMEOUT, OLLAMA_READ_TIMEOUT)
    last_err: BaseException | None = None
    for attempt in range(OLLAMA_HTTP_RETRIES + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except (ReadTimeout, Timeout) as e:
            last_err = e
            if attempt < OLLAMA_HTTP_RETRIES:
                wait = 5 * (attempt + 1)
                print(
                    f"  Ollama 응답 지연(타임아웃), {wait}초 후 재시도 "
                    f"({attempt + 1}/{OLLAMA_HTTP_RETRIES})…"
                )
                time.sleep(wait)
        except RequestsConnectionError as e:
            last_err = e
            if attempt < OLLAMA_HTTP_RETRIES:
                wait = 5 * (attempt + 1)
                print(
                    f"  Ollama 연결 실패, {wait}초 후 재시도 "
                    f"({attempt + 1}/{OLLAMA_HTTP_RETRIES})…"
                )
                time.sleep(wait)
    assert last_err is not None
    raise last_err


# ==============================
#  유틸리티 함수 (프롬프트 조립)
# ==============================


def build_big_categories_block(big_categories: list[dict] | None) -> str:
    """Worker `summary_prompts.build_big_categories_block` 과 동일 형식."""
    if not big_categories:
        return ""
    pairs: list[str] = []
    for c in big_categories:
        cid = c.get("id")
        name = (c.get("name", "") or "").strip()
        if cid is None:
            continue
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            continue
        pairs.append(f"{cid_int}: {name}")
    if not pairs:
        return ""
    return "[빅카테고리(bc_id) 선택 목록]\n" + ", ".join(pairs) + "\n\n"


def metadata_user_prompt(
    summary: str, bc_block: str, max_summary_chars: int = 4000
) -> str:
    """Worker `metadata_user_prompt` 와 동일 계약."""
    snip = (summary or "").strip()[:max_summary_chars]
    bc_fallback = (
        ""
        if bc_block
        else "주의: [빅카테고리 선택 목록]이 없으므로 둘째 줄은 반드시 'bc_id: 0'으로 출력하라.\n\n"
    )
    return (
        f"{bc_block}"
        f"{bc_fallback}"
        "지시: 아래 [대상 텍스트]에서 법률안·안건 제목을 찾아 반드시 첫 줄에 title: 로 출력하라. "
        "제목을 찾지 못했다고 비우지 말고, 텍스트 상단에서 가장 구체적인 명칭 한 줄을 title로 써라. "
        "요약에 '## 제목 (명칭)' 형태면 title에는 괄호 안 명칭만, 샵·마크다운·레이블은 넣지 말 것. "
        "sc_keyword는 쉼표로 구분한 짧은 단어 최대 2개만.\n\n"
        "[대상 텍스트]\n"
        f"{snip}\n"
    )


def clean_text(text: str) -> str:
    text = re.sub(r"[\n\t\r]+", " ", text)
    text = re.sub(r"([가-힣])\s{2,}([가-힣])", r"\1\2", text)
    text = re.sub(r"([가-힣])\s+([음함임됨])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 메타데이터 후처리 규칙 (문서 종류와 무관하게 동일 적용)
# ---------------------------------------------------------------------------
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
    # 흔한 어미만 음절 가운데 잘린 공백 (있 음 등)
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
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩị" r"òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]"
)

_SUMMARY_NOISE_RE = re.compile(
    r"[ぁ-ゖァ-ヺ々〆〤]|ことを|进行|same|etc\.|자sal|放送|공공机构",
    re.IGNORECASE,
)


def summary_should_use_extract_fallback(summary: str) -> bool:
    """R6·R7: 조문 베끼기·이종어 비율."""
    s = summary or ""
    if _SUMMARY_CONTAMINATION_RE.search(s):
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
    """
    요약 텍스트의 품질 점수(0~5). 점수가 높을수록 채택 우선.
    """
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


def summary_is_acceptable(summary: str) -> bool:
    """
    저장 가능한 최소 품질 기준.
    """
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
        t = re.sub(r"\s+", " ", t)
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


def finetune_input_snippet(text: str, max_chars: int) -> str:
    """R10: 학습 input — 길면 제안이유·주요내용 앵커 우선."""
    if len(text) <= max_chars:
        return text
    margin = FINETUNE_ANCHOR_MARGIN
    for anchor in ("제안이유", "제안 이유", "주요내용", "주요 내용"):
        idx = text.find(anchor)
        if idx != -1:
            start = max(0, idx - margin)
            chunk = text[start : start + max_chars]
            if len(chunk) >= min(max_chars, max(2000, max_chars // 2)):
                return chunk[:max_chars]
    return text[:max_chars]


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


def is_tl_summary_truncated_or_broken(s: str) -> bool:
    """R10: 특정 단어 나열 없이 길이·마침표 개수·끝 문장부호로만 판단."""
    t = (s or "").strip()
    if len(t) < RULES.summary_min_chars:
        return True
    periods = t.count(".") + t.count("。")
    if periods < RULES.summary_min_sentence_punct:
        return True
    if len(t) > 80 and not re.search(r"[.!?。…]['\"」\s]*$", t):
        return True
    return False


# ==============================
# 4. 제목 추출
# ==============================


def extract_title(text: str) -> str:
    # 1순위: ~법률안
    match = re.search(r"([가-힣\s]+법률안)", text[:300])
    if match:
        return normalize_title(match.group(1))
    return normalize_title(text[:80])


def normalize_title(title: str) -> str:
    title = title.replace("  ", " ").strip()
    title = re.sub(r"\s+", " ", title)
    return title


# ==============================
# 5. 키워드 추출
# ==============================


# 법률 제목에서 자주 나오는 의미 단위 패턴
SPLIT_PATTERNS = [
    "및",
    "에관한",
    "을위한",
    "의설치",
    "의운영",
    "의지정",
    "의보호",
    "의촉진",
    "의지원",
    "의강화",
    "의개선",
]


kiwi = Kiwi()


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
    text = re.sub(r"([가-힣]) ([가-힣])", r"\1\2", text)
    text = re.sub(r"([가-힣]) ([가-힣])", r"\1\2", text)

    split_rx = r"(?:제안\s*이유|제안이유)(?:및주요내용)?|주요\s*내용"
    parts = re.split(split_rx, text, maxsplit=1)
    if len(parts) < 2:
        snip = text[:800].strip()
        return snip if snip else text[:480]

    target = parts[-1]
    cutoff = re.search(r"발의자\s*:|법률제\s*호|부\s*칙|신ㆍ구조문|신·구조문", target)
    if cutoff:
        target = target[: cutoff.start()]

    target = re.sub(r"의안\s*번호.*$", "", target, flags=re.DOTALL)
    target = re.sub(r"\(안제[^)]+\)", "", target)
    target = re.sub(r"\s+\.", ".", target)
    target = re.sub(r"\s+", " ", target).strip()

    sentences = re.split(r"(?<=[음임함됨])\s*", target)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 20]

    if not sentences:
        return (target[:800] if target else text[:800]).strip() or text[:480]

    merged = " ".join(sentences[:4])
    return merged if len(merged) >= 80 else text[:800].strip() or merged


def format_metadata_block(
    title: str, bc_id: int, sc_keyword: str, tl_summary: str
) -> str:
    return (
        f"title: {title}\n"
        f"bc_id: {bc_id}\n"
        f"sc_keyword: {sc_keyword}\n"
        f"tl_summary: {tl_summary}"
    )


def build_record_for_pdf(
    source_pdf: str, text: str, bc_block: str
) -> tuple[dict[str, Any], str]:
    """
    한 PDF에 대해 DB용 dict와 학습용 output 블록 문자열을 생성.
    """
    best_meta_row: dict[str, Any] | None = None
    best_block = ""
    best_score = -1

    for _ in range(METADATA_QUALITY_RETRIES + 1):
        raw_output = generate_summary_ollama(text, bc_block)
        parsed = parse_llm_metadata_block(raw_output)

        title = reconcile_title_with_text((parsed.get("title") or "").strip(), text)
        bc_id = normalize_bc_id(parsed.get("bc_id"), text)
        kw_raw = (parsed.get("sc_keyword") or "").strip() or extract_keyword(title)
        keyword = scrub_sc_keyword(normalize_llm_output_text(kw_raw))
        if not keyword or not sc_keyword_plausible(keyword, text, title):
            keyword = scrub_sc_keyword(
                normalize_llm_output_text(extract_keyword(title))
            ) or normalize_llm_output_text(extract_keyword(title))

        summary = normalize_llm_output_text((parsed.get("tl_summary") or "").strip())
        summary = _strip_long_latin_tokens(summary)
        summary = loosen_dense_hangul_summary(summary)
        if summary_should_use_extract_fallback(summary):
            summary = normalize_llm_output_text(
                _strip_long_latin_tokens(
                    loosen_dense_hangul_summary(extract_summary(text))
                )
            )
        if (
            not summary
            or summary.count(".") + summary.count("。")
            < RULES.summary_min_sentence_punct
            or is_tl_summary_truncated_or_broken(summary)
        ):
            summary = extract_summary(text)
            summary = normalize_llm_output_text(
                _strip_long_latin_tokens(loosen_dense_hangul_summary(summary))
            )

        block = format_metadata_block(title, bc_id, keyword, summary)
        meta_row = {
            "source_pdf": source_pdf,
            "title": title,
            "bc_id": bc_id,
            "sc_keyword": keyword,
            "tl_summary": summary,
        }

        score = summary_quality_score(summary)
        if score > best_score:
            best_score = score
            best_meta_row = meta_row
            best_block = block
        if summary_is_acceptable(summary):
            return meta_row, block

    # 모든 재시도 실패 시에도 가장 점수 높은 결과를 반환
    if best_meta_row is not None:
        return best_meta_row, best_block
    raise RuntimeError("메타데이터 생성 실패: 유효한 요약을 만들지 못했습니다.")


def process_all() -> None:
    pdf_dir = PDF_DIR
    meta_path = METADATA_JSONL
    meta_rows: list[dict[str, Any]] = []
    finetune_rows: list[dict[str, Any]] = []

    bc_block = build_big_categories_block(DB_BIG_CATEGORIES)

    for file in sorted(os.listdir(pdf_dir)):
        if not file.endswith(".pdf"):
            continue

        try:
            path = os.path.join(pdf_dir, file)
            reader = PdfReader(path)
            text = " ".join(
                p.extract_text() or "" for p in reader.pages if p.extract_text()
            )
            text = clean_text(text)

            if len(text) < 100:
                continue

            meta_row, output_block = build_record_for_pdf(file, text, bc_block)
            meta_rows.append(meta_row)

            if WRITE_FINETUNE:
                finetune_rows.append(
                    {
                        # Worker 추론 시 system 자리와 동일 문자열 → LoRA가 현장 계약과 맞춤
                        "instruction": METADATA_SYSTEM,
                        "input": finetune_input_snippet(text, FINETUNE_INPUT_MAX_CHARS),
                        "output": output_block,
                    }
                )

            print(f"완료: {file}")

        except Exception as e:
            print(f"에러: {file}, {e}")

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


if __name__ == "__main__":
    process_all()
