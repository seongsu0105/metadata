import os
import json
import re
from PyPDF2 import PdfReader

# ==============================
# 1. 설정 및 프롬프트 정의
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


def assign_bc_id(text):
    """텍스트 내 키워드를 분석하여 적절한 bc_id 반환"""
    category_map = {
        1: ["정치", "행정", "지방자치", "공무원", "선거"],
        2: ["경제", "산업", "금융", "조세", "세법", "기업", "공정거래"],
        3: ["복지", "장애인", "보건", "의료", "노동", "가족"],
        4: ["법사", "사법", "형법", "민법", "범죄", "검찰", "경찰", "안전"],
        5: ["교육", "문화", "콘텐츠", "예술", "체육", "관광", "언론", "방송"],
        6: ["과학", "기술", "정보통신", "인공지능", "데이터", "원자력"],
        7: ["국방", "외교", "통일", "군사", "안보"],
        8: ["환경", "에너지", "기후", "자원", "폐기물"],
        9: ["농림", "축산", "식품", "해양", "수산"],
        10: ["국토", "교통", "주택", "건설", "도로", "철도", "항공"],
    }

    for bc_id, keywords in category_map.items():
        if any(kw in text for kw in keywords):
            return bc_id
    return 1  # 기본값


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
# 2. 유틸리티 함수 (프롬프트 조립)
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
# 3. 데이터 추출 및 정제 로직 (품질 향상용)
# ==============================


def clean_text(text):
    # 1. 줄 바꿈(\n)과 탭(\t)을 모두 일반 공백으로 변경
    text = re.sub(r"[\n\t\r]+", " ", text)

    # 2. 핵심 해결책: 한글 글자 사이의 2개 이상 공백을 1개로 축소
    # (또는 아예 붙여버리고 싶다면 ''로 대체 가능)
    text = re.sub(r"([가-힣])\s{2,}([가-힣])", r"\1\2", text)

    # 3. 문장 끝 '있  음' 처럼 조사/어미가 떨어진 경우 붙이기
    text = re.sub(r"([가-힣])\s+([음함임됨함])", r"\1\2", text)

    # 4. 마침표 뒤에 공백이 없는 경우 강제 부여 (요약 가독성)
    text = re.sub(r"\.(?=[가-힣])", ". ", text)

    # 5. 연속된 공백을 하나로 합치고 양끝 공백 제거
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_logic_title(text):
    match = re.search(r"([가-힣\s\(\)·]+일부개정법률안)", text[:300])
    if match:
        title = match.group(1).strip()
        return " ".join(title.split())

    match = re.search(r"([가-힣A-Za-z0-9\s]+(?:법률안|안건))\s?\(", text[:300])
    if match:
        return " ".join(match.group(1).split())

    raw_head = text[:100].split("제안이유")[0].strip()
    return " ".join(raw_head.split()[:5])


def extract_logic_keyword(title):
    # '일부개정법률안' 제거
    keyword = title.replace("일부개정법률안", "").strip()

    # 끝에 붙은 조사(~및, ~에 관한, ~등) 제거
    keyword = re.sub(r"(및|에관한|등|의|한)$", "", keyword).strip()

    # 너무 길면 핵심 단어만 (예: 정보통신망법)
    if len(keyword) > 8:
        # '법'으로 끝나면 그 앞까지만 살리거나 핵심구 추출
        keyword = keyword[:8]

    return f"{keyword}, 개정안" if keyword else "개정안"


def extract_logic_summary(text, max_len=300):
    # '제안이유' 혹은 '제안이유 및 주요내용' 이후 텍스트 추출
    parts = re.split(r"제안이유(?:및주요내용)?", text)
    target_text = parts[-1].strip()

    # 노이즈(의안번호, 발의자 등) 제거
    target_text = re.split(r"의안\s?번호|발의\s?연월일|발의자\s?:", target_text)[
        0
    ].strip()

    sentences = re.findall(r"[^.!?]*[.!?]", target_text)

    summary_list = []
    current_len = 0

    for s in sentences[:3]:
        s_clean = s.strip()
        if current_len + len(s_clean) <= max_len:
            summary_list.append(s_clean)
            current_len += len(s_clean) + 1
        else:
            break

    return " ".join(summary_list) if summary_list else target_text[:max_len]


# ==============================
# 4. 실행 프로세스
# ==============================


def process_all():
    pdf_dir = "pdfdata"
    output_path = "dataset.jsonl"
    bc_block = build_big_categories_block(DB_BIG_CATEGORIES)

    if not os.path.exists(pdf_dir):
        print("PDF 폴더 없음")
        return

    results = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            try:
                path = os.path.join(pdf_dir, file)
                reader = PdfReader(path)
                full_text = clean_text(
                    " ".join(
                        [p.extract_text() for p in reader.pages if p.extract_text()]
                    )
                )

                if len(full_text) < 100:
                    continue

                title = extract_logic_title(full_text)
                bc_id = assign_bc_id(full_text)
                sc_keyword = extract_logic_keyword(title)
                summary = extract_logic_summary(full_text)

                output_content = (
                    f"title: {title}\n"
                    f"bc_id: {bc_id}\n"
                    f"sc_keyword: {sc_keyword}\n"
                    f"tl_summary: {summary}"
                )

                results.append(
                    {
                        "instruction": "당신은 법률 원문을 분석하여 요약·메타데이터를 생성하는 AI입니다. 매 요청마다 system으로 전달되는 지시와 출력 형식을 그대로 따르세요.",
                        "input": metadata_user_prompt(full_text, bc_block),
                        "output": output_content,
                    }
                )
                print(f"처리 완료: {file}")
            except Exception as e:
                print(f"에러: {file} ({e})")

    with open(output_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"\n 총 {len(results)}개의 데이터셋이 생성되었습니다.")


if __name__ == "__main__":
    process_all()
