import os
import json
import re
import time
from typing import Any
import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ReadTimeout, Timeout
from PyPDF2 import PdfReader
from kiwipiepy import Kiwi

# ---------------------------------------------------------------------------
# 산출물
# - metadata.jsonl: DB·배포용 순수 메타데이터 (본 역할)
# - finetune_dataset.jsonl: instruction/input/output 학습용 (선택)
# ---------------------------------------------------------------------------
METADATA_JSONL = "metadata.jsonl"
FINETUNE_JSONL = "finetune_dataset.jsonl"
WRITE_FINETUNE = True

PDF_DIR = "pdfdata"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
# title~tl_summary 4줄 + 세 문장 요약이 중간에 끊기지 않도록 여유 있게 설정
OLLAMA_NUM_PREDICT = 1536
# Ollama가 응답을 끝낼 때까지 기다리는 최대 시간(초).
# 짧으면 Read timed out — PDF가 길거나 GPU가 느리면 300초를 넘기기 쉽다.
OLLAMA_CONNECT_TIMEOUT = 30
OLLAMA_READ_TIMEOUT = 900
# 타임아웃·일시 연결 실패 시 재시도 횟수(추가)
OLLAMA_HTTP_RETRIES = 2

METADATA_SYSTEM = (
    "당신은 법률 원문을 분석하여 요약·메타데이터를 생성하는 AI입니다. 매 요청마다 system으로 전달되는 지시와 출력 형식을 그대로 따르세요."
    "역할: [대상 텍스트]만 보고 DB 저장용 메타데이터를 키:값 한 줄씩 출력한다.\n\n"
    "[절대 순서 — 반드시 이 순서로 4줄만 출력]\n"
    "1) 첫 줄: title: (실제 제목 문자열)\n"
    "2) 둘째 줄: bc_id: (숫자)\n"
    "3) 셋째 줄: sc_keyword: (핵심 단어 최대 2개, 반드시 쉼표로만 구분. 예: 전력, 송전)\n"
    "4) 넷째 줄: tl_summary: (세 문장, 한 줄로)\n\n"
    "[title — 최우선·생략 절대 금지]\n"
    '- title 값은 빈 칸·"제목 없음"·"N/A"·"해당 없음"·"미정" 금지. 반드시 구체적인 법률안·안건 명칭을 한글로 채운다.\n'
    "- 찾는 순서: (가) '## 제목' 또는 '1. ## 제목' 바로 아래 한 줄 (나) '의안명'·'법률안 명' 근처 문장 (다) 첫머리에 나온 안건명·법령명 전체 (라) 그래도 없으면 문서 맨 앞 80자 이내 핵심 명사구를 붙여 하나의 제목 문장으로 만든다(빈 값 금지).\n"
    "- 요약에 '## 제목 (○○법률안)'처럼 나오면 title 값은 괄호 안 ○○만 넣는다. '## 제목', '#', '제목' 레이블·괄호 자체는 title에 넣지 않는다.\n"
    "- 마크다운 기호(##, **)는 title 값에 넣지 말고 법률안·안건 공식 명칭 문장만 한 줄로.\n\n"
    "[bc_id]\n"
    "- 반드시 시스템에 붙은 [대분류(bc_id) 선택 목록]에 나온 id(1~10)만 사용한다.\n"
    "- 의안번호·법령번호·연도 등 다른 숫자를 bc_id에 넣지 말 것.\n"
    "- 목록에 없는 숫자·임의 숫자 금지.\n\n"
    "[sc_keyword]\n"
    "- 법안이 직접 다루는 대상(직군·계층·제도명 등) 단어 하나.\n\n"
    "- 반드시 '조사(~및, ~에 관한, ~등)'를 제거한 완성된 명사형으로 출력하라.\n"
    "- 예: '공예문화산업진흥법및' (X) -> '공예문화산업법' (O)\n\n"
    "[tl_summary]\n"
    "- 반드시 세 문장(마침표 3개 이상). 각 문장은 주어·서술어를 갖춘 **완결 문장**으로 끝낸다. 중간에 끊기지 않게 한다.\n"
    "- ① 개정 취지(무엇이 어떻게 바뀌는지) ② 문제 인식 또는 개정 전·후 ③ 시행일·기대 효과·정리 중 하나로 마무리.\n"
    "- **출력 문자**: 한글·숫자·공백·마침표·쉼표·가운뎃점(ㆍ) 위주. 영단어·독일어 등 라틴 문자, 일본어, 중국어 **간체·한자 단독 표기 금지**(법 용어는 한글로 풀어쓴다. 예: 죄, 퇴직, 규정).\n"
    "- 조문 번호 나열·'(생략)'·'∼' 위주 인용·정의 조항 베끼기 금지. 법안 **이유·취지** 수준으로만 요약한다.\n"
    "- 띄어쓰기 규칙을 지켜 읽기 쉬운 문장으로 쓴다. 글자를 붙여 한 줄로만 쓰지 말 것.\n"
    "- 문장 내 '있 음', '개 정 안', '배 우자' 같이 어색하게 끼인 공백 금지.\n"
    '"요약 없음"·한 문장만·빈 값 금지.\n\n'
    "[공통]\n"
    "- 설명·인사·JSON·코드펜스·추가 줄 금지. 위 4키만, 키 이름 철자 정확히 title, bc_id, sc_keyword, tl_summary.\n"
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

FINETUNE_INSTRUCTION = (
    "당신은 법률 원문을 분석하여 요약·메타데이터를 생성하는 AI입니다. "
    "매 요청마다 system으로 전달되는 지시와 출력 형식을 그대로 따르세요."
)


def assign_bc_id(text: str) -> int:
    category_map = {
        1: ["정치", "행정", "지방자치", "공무원", "선거", "자치단체"],
        2: ["경제", "산업", "금융", "조세", "세법", "기업", "공정거래", "재정", "조세특례", "상속세", "증여세", "자본시장", "펀드"],
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
        5: ["교육", "문화", "콘텐츠", "예술", "체육", "관광", "언론", "방송", "박물관", "미술관", "공예", "대중문화"],
        6: ["과학", "기술", "정보통신", "인공지능", "데이터", "원자력", "정보보호", "정보통신망"],
        7: ["국방", "외교", "통일", "군사", "안보"],
        8: ["환경", "에너지", "기후", "자원", "폐기물"],
        9: ["농림", "축산", "식품", "해양", "수산"],
        10: ["국토", "교통", "주택", "건설", "도로", "철도", "항공", "소방", "하도급", "계약"],
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
    for line in text.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        m = key_pat.match(line)
        if m:
            k = m.group(1).lower()
            out[k] = m.group(2).strip()
            continue
        if ": " in line:
            k, _, v = line.partition(": ")
            key = k.strip()
            if key in ("title", "bc_id", "sc_keyword", "tl_summary") and key not in out:
                out[key] = v.strip()
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
    if not big_categories:
        return ""
    pairs = [
        f"{int(c['id'])}: {c['name']}"
        for c in big_categories
        if c.get("id") is not None
    ]
    return "[대분류(bc_id) 선택 목록]\n" + ", ".join(pairs) + "\n\n"


def metadata_user_prompt(
    summary: str, bc_block: str, max_summary_chars: int = 4000
) -> str:
    snip = (summary or "").strip()[:max_summary_chars]
    return (
        f"{bc_block}"
        "지시: 아래 [대상 텍스트]에서 법률안·안건 제목을 찾아 반드시 첫 줄에 title: 로 출력하라. "
        "bc_id는 위 목록의 1~10 중 하나만. 의안번호·다른 숫자는 bc_id에 쓰지 말 것. "
        "네 번째 줄 tl_summary까지 반드시 출력하라. "
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


# LLM 출력에 섞이기 쉬운 오탈·이종 문자 (긴 것부터)
_SUBSTR_FIXES: tuple[tuple[str, str], ...] = (
    ("자sal위해", "자살 위해"),
    ("자Sal위해", "자살 위해"),
    ("제19조삭 제", "제19조 삭제"),
    ("없음 이분명한", "없음이 분명한"),
    ("임 기를", "임의를"),
    ("1 00분의", "100분의"),
    (" 입찰 참 gia ", " 입찰 참가 "),
    ("참 gia ", "참가 "),
    (" gia ", " 가 "),
    (" address한다", " 보완한다"),
    ("Regel", "규정"),
    ("regel", "규정"),
    ("배 우자", "배우자"),
    ("수급권 자", "수급권자"),
)

_HANZI_TO_HANGUL: tuple[tuple[str, str], ...] = (
    ("위헌결定的", "위헌결정적"),
    ("규定", "규정"),
    ("权리", "권리"),
    ("退직", "퇴직"),
    ("除く", "제외하고"),
    ("规定", "규정"),
    ("罪", "죄"),
    ("权", "권"),
)

# 요약이 생성 도중 잘린 것으로 보일 때(폴백: extract_summary)
_TL_SUMMARY_BAD_ENDINGS: tuple[str, ...] = (
    "근거",
    "패러다임",
    "적시",
    "연기함",
    "하고있어임",
    "미비하여해임",
    "운송사",
    "개인으로규정하고있어임",
    "삭제ㆍ임",
)


def normalize_llm_output_text(s: str) -> str:
    """모델·OCR 혼입 문자·흔한 오타를 한국어 쪽으로 정리."""
    if not s:
        return s
    t = s.replace("\ufeff", "")
    for a, b in _SUBSTR_FIXES:
        t = t.replace(a, b)
    for a, b in _HANZI_TO_HANGUL:
        t = t.replace(a, b)
    # 조사 앞 불필요 공백 (예: 해임 을 → 해임을)
    t = re.sub(
        r"([가-힣]{2,})\s+(을|를|이|가|은|는|과|와)(?=[가-힣\s]|$)",
        r"\1\2",
        t,
    )
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def loosen_dense_hangul_summary(s: str) -> str:
    """공백이 거의 없는 '벽돌' 한글 요약을 형태소 경계로 띄어쓴다."""
    if not s or len(s) < 35:
        return s
    if (s.count(" ") / len(s)) > 0.045:
        return s
    try:
        return " ".join(tok.form for tok in kiwi.tokenize(s))
    except Exception:
        return s


def is_tl_summary_truncated_or_broken(s: str) -> bool:
    t = (s or "").strip()
    if len(t) < 40:
        return True
    periods = t.count(".") + t.count("。")
    if periods < 2:
        return True
    core = t.rstrip(". …")
    for end in _TL_SUMMARY_BAD_ENDINGS:
        if core.endswith(end):
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

    parts = re.split(r"제안이유(?:및주요내용)?", text)
    if len(parts) < 2:
        return text[:480]

    target = parts[1]
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
        return target[:480]

    return " ".join(sentences[:3])


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
    raw_output = generate_summary_ollama(text, bc_block)
    parsed = parse_llm_metadata_block(raw_output)

    title = normalize_llm_output_text(
        (parsed.get("title") or "").strip() or extract_title(text)
    )
    bc_id = normalize_bc_id(parsed.get("bc_id"), text)
    keyword = normalize_llm_output_text(
        (parsed.get("sc_keyword") or "").strip() or extract_keyword(title)
    )
    summary = normalize_llm_output_text((parsed.get("tl_summary") or "").strip())
    summary = loosen_dense_hangul_summary(summary)
    if (
        not summary
        or summary.count(".") + summary.count("。") < 2
        or is_tl_summary_truncated_or_broken(summary)
    ):
        summary = extract_summary(text)
        summary = normalize_llm_output_text(loosen_dense_hangul_summary(summary))

    block = format_metadata_block(title, bc_id, keyword, summary)
    meta_row = {
        "source_pdf": source_pdf,
        "title": title,
        "bc_id": bc_id,
        "sc_keyword": keyword,
        "tl_summary": summary,
    }
    return meta_row, block


def process_all() -> None:
    pdf_dir = PDF_DIR
    meta_path = METADATA_JSONL
    meta_rows: list[dict[str, Any]] = []
    finetune_rows: list[dict[str, Any]] = []

    bc_block = build_big_categories_block(DB_BIG_CATEGORIES)

    for file in os.listdir(pdf_dir):
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
                        "instruction": FINETUNE_INSTRUCTION,
                        "input": text[:2000],
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
