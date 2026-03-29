import os
import json
import re
from typing import Any
import requests
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
# 4줄 전부 생성 전에 잘리지 않도록 여유 있게 설정
OLLAMA_NUM_PREDICT = 768

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
    "- 반드시 세 문장(마침표 3개 이상). ①개정 결론(무엇이 어떻게 바뀌는지) ②변경 전·후 ③시행일 또는 기대 효과.\n"
    "- 문장 내 '있 음', '개 정 안'과 같은 비정상적 공백은 반드시 제거하여 자연스러운 문장으로 작성하라.\n"
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
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "system": METADATA_SYSTEM,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": OLLAMA_NUM_PREDICT,
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


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

    title = (parsed.get("title") or "").strip() or extract_title(text)
    bc_id = normalize_bc_id(parsed.get("bc_id"), text)
    keyword = (parsed.get("sc_keyword") or "").strip() or extract_keyword(title)
    summary = (parsed.get("tl_summary") or "").strip()
    if not summary or summary.count(".") < 2:
        summary = extract_summary(text)

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
