import os
import json
import re
from PyPDF2 import PdfReader
from kiwipiepy import Kiwi

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

# ==============================
# 2. bc_id
# ==============================


def assign_bc_id(text):
    category_map = {
        1: ["정치", "행정", "지방자치", "공무원", "선거"],
        2: ["경제", "산업", "금융", "조세", "세법", "기업", "공정거래"],
        3: ["복지", "장애인", "보건", "의료", "노동", "가족"],
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
        ],
        5: ["교육", "문화", "콘텐츠", "예술", "체육", "관광", "언론", "방송"],
        6: ["과학", "기술", "정보통신", "인공지능", "데이터", "원자력"],
        7: ["국방", "외교", "통일", "군사", "안보"],
        8: ["환경", "에너지", "기후", "자원", "폐기물"],
        9: ["농림", "축산", "식품", "해양", "수산"],
        10: ["국토", "교통", "주택", "건설", "도로", "철도", "항공"],
    }

    scores = {k: 0 for k in category_map}

    for bc_id, keywords in category_map.items():
        for kw in keywords:
            scores[bc_id] += text.count(kw)

    return max(scores, key=scores.get)


METADATA_SYSTEM = (
    "역할: [대상 텍스트]만 보고 DB 저장용 메타데이터를 키:값 한 줄씩 출력한다.\n\n"
    "[절대 순서 — 반드시 이 순서로 4줄만 출력]\n"
    "1) 첫 줄: title: (실제 제목 문자열)\n"
    "2) 둘째 줄: bc_id: (숫자)\n"
    "3) 셋째 줄: sc_keyword: (핵심 단어 최대 2개, 반드시 쉼표로만 구분. 예: 전력, 송전)\n"
    "4) 넷째 줄: tl_summary: (세 문장, 한 줄로)\n\n"
    "[title — 최우선·생략 절대 금지]\n"
    "- 출력의 반드시 첫 번째 줄은 title: 로 시작해야 한다.\n"
    '- title 값은 빈 칸·"제목 없음"·"N/A"·"해당 없음"·"미정" 금지. 반드시 구체적인 법률안·안건 명칭을 한글로 채운다.\n'
    "- 찾는 순서: (가) '## 제목' 또는 '1. ## 제목' 바로 아래 한 줄 (나) '의안명'·'법률안 명' 근처 문장 (다) 첫머리에 나온 안건명·법령명 전체 (라) 그래도 없으면 문서 맨 앞 80자 이내 핵심 명사구를 붙여 하나의 제목 문장으로 만든다(빈 값 금지).\n"
    "- 요약에 '## 제목 (○○법률안)'처럼 나오면 title 값은 괄호 안 ○○만 넣는다. '## 제목', '#', '제목' 레이블·괄호 자체는 title에 넣지 않는다.\n"
    "- 마크다운 기호(##, **)는 title 값에 넣지 말고 법률안·안건 공식 명칭 문장만 한 줄로.\n\n"
    "[bc_id]\n"
    "- [빅카테고리(bc_id) 선택 목록]에 있는 id 숫자만. 목록에 없는 숫자·임의 숫자 금지. 애매하면 목록 중 가장 가까운 하나.\n\n"
    "[sc_keyword]\n"
    "- 법안이 직접 다루는 대상(직군·계층·제도명 등) 단어 하나.\n\n"
    "- 반드시 '조사(~및, ~에 관한, ~등)'를 제거한 완성된 명사형으로 출력하라.\n"
    "- 예: '공예문화산업진흥법및' (X) -> '공예문화산업법' (O)\n\n"
    "[tl_summary]\n"
    "- 반드시 세 문장(마침표 3개 이상). ①개정 결론(무엇이 어떻게 바뀌는지) ②변경 전·후 ③시행일. "
    "- 문장 내 '있 음', '개 정 안'과 같은 비정상적 공백은 반드시 제거하여 자연스러운 문장으로 작성하라.\n"
    "- 내용은 ①현행법의 한계 ②개정안의 핵심 내용 ③기대 효과를 포함하여 세 문장으로 구성하라.\n"
    '"요약 없음"·한 문장만·빈 값 금지.\n\n'
    "[공통]\n"
    "- 설명·인사·JSON·코드펜스·추가 줄 금지. 위 4키만, 키 이름 철자 정확히 title, bc_id, sc_keyword, tl_summary.\n"
)


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
        "제목을 찾지 못했다고 비우지 말고, 텍스트 상단에서 가장 구체적인 명칭 한 줄을 title로 써라. "
        "요약에 '## 제목 (명칭)' 형태면 title에는 괄호 안 명칭만, 샵·마크다운·레이블은 넣지 말 것. "
        "sc_keyword는 쉼표로 구분한 짧은 단어 최대 2개만.\n\n"
        "[대상 텍스트]\n"
        f"{snip}\n"
    )


# ==============================
# 3. 텍스트 정제 (강화)
# ==============================


def clean_text(text):
    text = re.sub(r"[\n\t\r]+", " ", text)

    # 한글 띄어쓰기 복구
    text = re.sub(r"([가-힣])\s{2,}([가-힣])", r"\1\2", text)

    # 잘린 조사 복구
    text = re.sub(r"([가-힣])\s+([음함임됨])", r"\1\2", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==============================
# 4. 제목 추출
# ==============================


def extract_title(text):
    # 1순위: ~법률안
    match = re.search(r"([가-힣\s]+법률안)", text[:300])
    if match:
        return normalize_title(match.group(1))

    # fallback
    return normalize_title(text[:80])


def normalize_title(title):
    title = title.replace("  ", " ").strip()

    # "일부개정법률안" 유지
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


def extract_keyword(title):
    # 1. 접미사 제거 (긴 것부터)
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

    # 2. 공백 기준 분리 먼저 시도
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

    # 3. 공백 분리 실패 → kiwi 형태소 분석
    word = refined_parts[0] if refined_parts else clean_title
    tokens = kiwi.tokenize(word)
    nouns = [
        token.form
        for token in tokens
        if token.tag in ("NNG", "NNP") and len(token.form) >= 2
    ]

    if len(nouns) >= 2:
        return f"{nouns[0]}, {nouns[1]}"
    elif len(nouns) == 1:
        return nouns[0]

    # 4. fallback → half
    if len(word) <= 3:
        return word
    half = len(word) // 2
    return f"{word[:half]}, {word[half:]}"


# ==============================
# 6. 요약
# ==============================


def extract_summary(text):
    # 1. 페이지 구분자 먼저 제거
    text = re.sub(r"-\s*\d+\s*-", " ", text)

    # 2. 비정상 공백 제거
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([가-힣]) ([가-힣])", r"\1\2", text)
    text = re.sub(r"([가-힣]) ([가-힣])", r"\1\2", text)

    # 3. 제안이유 섹션 추출
    parts = re.split(r"제안이유(?:및주요내용)?", text)
    if len(parts) < 2:
        return text[:300]

    target = parts[1]

    # 4. 발의자/조문 시작 전까지 자르기
    cutoff = re.search(r"발의자\s*:|법률제\s*호|부\s*칙|신ㆍ구조문|신·구조문", target)
    if cutoff:
        target = target[: cutoff.start()]

    # 5. 의안번호 + 이후 잔여 정보 제거 (끝까지)
    target = re.sub(r"의안\s*번호.*$", "", target, flags=re.DOTALL)
    # 안제XX조 형태 괄호 내용 제거
    target = re.sub(r"\(안제[^)]+\)", "", target)

    # 6. 공백+점 정리
    target = re.sub(r"\s+\.", ".", target)
    target = re.sub(r"\s+", " ", target).strip()

    # 7. 문장 분리
    sentences = re.split(r"(?<=[음임함됨])\s*", target)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 20]

    if not sentences:
        return target[:300]

    return " ".join(sentences[:3])


# ==============================
# 7. 실행
# ==============================


def process_all():
    pdf_dir = "pdfdata"
    output_path = "dataset.jsonl"

    results = []

    for file in os.listdir(pdf_dir):
        if not file.endswith(".pdf"):
            continue

        try:
            path = os.path.join(pdf_dir, file)
            reader = PdfReader(path)

            text = " ".join(
                [p.extract_text() for p in reader.pages if p.extract_text()]
            )

            text = clean_text(text)

            if len(text) < 100:
                continue

            title = extract_title(text)
            bc_id = assign_bc_id(text)
            keyword = extract_keyword(title)
            summary = extract_summary(text)

            output = (
                f"title: {title}\n"
                f"bc_id: {bc_id}\n"
                f"sc_keyword: {keyword}\n"
                f"tl_summary: {summary}"
            )

            results.append(
                {
                    "instruction": "당신은 법률 원문을 분석하여 요약·메타데이터를 생성하는 AI입니다. 매 요청마다 system으로 전달되는 지시와 출력 형식을 그대로 따르세요.",
                    "input": text[:2000],
                    "output": output,
                }
            )

            print(f"완료: {file}")

        except Exception as e:
            print(f"에러: {file}, {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n총 {len(results)}개 생성 완료")


if __name__ == "__main__":
    process_all()
