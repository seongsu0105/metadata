# =============================================================================
#   app/prompts/summary.py
# =============================================================================

"""
의안 요약·메타데이터 프롬프트 (Worker 전담).
메인은 OCR·권한 등 데이터만 넘기고, system/user 문자열은 여기서 조립한다.
`docs/ex/summary_prompts.py`와 동일 계약.
"""

BASE_SYSTEM_COMMON = (
    "공통 원칙:\n"
    "- 제공된 텍스트 근거 중심으로 작성\n"
    "- 수치·날짜·고유명사는 원문 표기 유지\n"
    "- 불명확한 정보는 '원문상 명시 없음'으로 표기\n"
    "- 결과 본문만 간결하게 출력\n"
)


# ----- 단문·전체 요약 (3~10페이지 등 한 번에 넣는 경로) -----
SUMMARY_SYSTEM_FINAL = (
    BASE_SYSTEM_COMMON
    + "역할: 국회 의안 통합 요약관.\n"
    + "법률 원문/구간 메모를 통합해 정책 검토형 요약문을 작성한다.\n"
    + "분량은 원문데이터 크기에 따라서 400~1000자 권장.\n"
    + "조문·부칙을 통째로 베끼지 말고 쟁점·변경 요지만 균형 있게 서술한다.\n"
)


def summary_user_single_document(full_text: str) -> str:
    """단문 요약(원문 통째 1회)."""
    return (
        "[법률 원문]\n"
        f"{full_text}\n\n"
        "위 원문을 6단원 순서대로 요약하세요.\n"
        "2번 단원에는 의안번호·발의연월일·발의자를 포함하고,\n"
        "3~6번 단원에는 제도 설계, 집행 영향, 시행 정보를 균형 있게 반영하세요."
    )


# ----- 장문 구간별(부분 청킹) -----
SUMMARY_SYSTEM_PARTIAL = (
    BASE_SYSTEM_COMMON
    + "역할: 장문 의안의 구간별 사실 추출관.\n"
    + "현재 구간에서 최종 요약에 필요한 근거 bullet만 추출한다.\n"
    + "출력 형식은 bullet(-) 목록으로 통일하고, 짧고 밀도 있게 작성한다.\n"
    + "설명 문구(예: '아래는 ... 정리한 내용입니다', '[참고: ...]')는 출력하지 않는다.\n"
)


def summary_user_partial(
    chunk_text: str,
    part_index: int,
    part_total: int,
    *,
    provenance: str | None = None,
) -> str:
    """구간 요약용 사용자 프롬프트. `provenance`는 합본 원문 기준 문자 구간·PDF 페이지(가능 시)."""
    meta = f"[청킹 좌표: {provenance}]\n" if provenance and str(provenance).strip() else ""
    return (
        f"{meta}[원문 일부: 전체 중 {part_index}/{part_total}번째 구간입니다]\n"
        f"{chunk_text}\n\n"
        "이 구간에서 새롭게 확인되는 사실을 bullet(-)로 정리하세요.\n"
        "아래 우선순위에 따라 최대 4~6개 bullet만 작성하세요.\n"
        "1) 식별정보: 제목, 의안번호, 발의자, 발의일\n"
        "2) 목적·배경: 제안이유, 문제 상황, 정책 목표\n"
        "3) 제도내용: 개정 조문, 대상, 절차, 권한, 범위\n"
        "4) 시행요소: 부칙, 시행일, 경과규정, 신구 차이\n"
        "각 bullet은 한 줄(가능하면 80자 이내)로 작성하고, 핵심 명사/동사 중심으로 압축하세요.\n"
        "bullet은 '주체-행위-대상-조건/시점' 구조를 우선 사용하고, 조문 번호/수치/기간은 원문 표기를 유지하세요.\n"
        "전문 용어(예: 증권금융회사, 과징금) 단어 중간에 불필요한 공백을 넣지 말고 자연스러운 한국어 문장으로 쓰세요.\n"
        "중복되는 식별정보는 1개 bullet로 묶고, 새 정보가 없으면 해당 항목은 생략하세요."
    )


# ----- 조각 메모 합성 → 최종 6단원 -----
def summary_user_final_from_chunk_memos(combined_memos: str) -> str:
    """구간 메모 종합 → 최종 6단원."""
    return (
        "[구간별 메모]\n"
        f"{combined_memos}\n\n"
        "메모를 종합해 6단원 ## 요약을 작성하세요.\n"
        "요약은 700~1100자, 각 단원 2~4문장으로 구성하세요.\n"
        "핵심 구성: 문제의 실체, 제도 설계, 이해관계 쟁점, 시행/집행 영향, 결론.\n"
        "메모 간 정보가 다르면 더 구체적인 조문/수치/시점을 우선 반영하세요.\n"
        "주의: 본문 앞뒤에 설명성 문구(예: '아래는 ...', '[참고: ...]')를 넣지 말고 바로 요약만 출력하세요."
    )


# ----- 메타데이터 (키:값 한 줄씩) -----
METADATA_SYSTEM = (
    BASE_SYSTEM_COMMON
    + "역할: 아래 사용자 메시지의 「대상 텍스트」만 보고 DB 저장용 메타데이터를 키:값 한 줄씩 출력한다.\n"
    + "출력 목표는 검색 인덱싱 품질 최적화다.\n\n"
    "[절대 순서 — 반드시 이 순서로 4줄만 출력]\n"
    "1) 첫 줄: title: (실제 제목 문자열)\n"
    "2) 둘째 줄: bc_id: (숫자)\n"
    "3) 셋째 줄: sc_keyword: (핵심 단어 최대 2개, 반드시 쉼표로만 구분)\n"
    "4) 넷째 줄: tl_summary: (세 문장, 한 줄로 — 원인·결과가 드러나게)\n\n"
    "— title 줄 —\n"
    "- 첫 줄은 title: 로 시작하고, 구체적인 법률안/안건 명칭만 쓴다.\n"
    "- 빈 값·'제목 없음'·N/A·'미정' 금지. 못 찾겠어도 「대상 텍스트」 상단에서 가장 구체적인 안건명·명사구 한 줄을 title로 채운다.\n"
    "- 우선순위: '## 제목' 또는 번호 목차 바로 아래 한 줄 → 의안명/법률안명 문장 → 상단 핵심 명사구.\n"
    "- 요약·원문에 '## 제목 (○○법률안)' 형태면 title 값은 괄호 안 명칭 중심으로 짓고, '#', '##', '제목' 레이블은 title 값에 넣지 않는다.\n"
    "- title 값 끝에 [bc_id], (bc_id), bc_id 등 플레이스홀더·꼬리표·대괄호 태그를 절대 붙이지 않는다.\n\n"
    "— bc_id 줄 —\n"
    "- 사용자 메시지에 「대분류 후보 목록」이 있으면, 그 목록에 나온 id 숫자 중 요약 주제와 이름이 가장 잘 맞는 하나만 고른다.\n"
    "- 목록에 없는 id·임의 숫자 금지. 애매하면 후보 이름과 「대상 텍스트」 용어를 대조해 가장 가까운 하나만 고른다.\n\n"
    "— sc_keyword 줄 —\n"
    "- 「대상 텍스트」·안건 제목에 실제로 등장하는 단어만 쓴다.\n"
    "- 법안 대상·제도·쟁점을 드러내는 짧은 명사구 최대 2개, 쉼표로만 구분. 지나치게 일반적인 단어(정책, 법률, 개선만 등)만으로 채우지 않는다.\n"
    "- 필요 없는 조사·'~및' 꼬리를 붙인 미완성 명사(예: ○○법및)는 쓰지 말고 완성된 명사형으로 정리한다.\n\n"
    "— tl_summary 줄 —\n"
    "- 세 완결 문장(마침표 최소 3개 권장). ① 배경·쟁점 ② 개정·제도 내용·효과 ③ 시행·시점·파급 순으로 원인-결과가 읽히게.\n"
    "- title 줄·검토보고서 표제·`[법안명]` 헤더를 그대로 붙여넣지 않는다. 세 문장은 「대상 텍스트」 근거만으로 쓴다.\n"
    "- 「대상 텍스트」에 '국회 의안 통합 요약', '1. 문제의 실체', '###' 같은 목차·마크다운이 있어도 tl_summary에 복사하지 않는다.\n"
    "- 조문 전문 인용·'(생략)'·정의 조항만 잘라 붙이지 말고 제안 취지 수준으로 압축한다.\n"
    "- 완성형 한글 위주. 음절과 받침(종성) 사이에 불필요한 공백 넣지 않는다.\n"
    "- tl_summary 값은 한 줄만. `#`, `###`, 번호 목차(1. 2.) 금지. 각 문장은 마침표로 끝낸다.\n\n"
    "— 형식 —\n"
    "- 위 4줄만 출력하고 키 이름은 title, bc_id, sc_keyword, tl_summary로 고정한다.\n"
    "- 설명 문단·마크다운 제목·빈 줄 없이 바로 title: 부터 출력한다.\n"
)


def _metadata_snippet_from_summary(text: str, max_chars: int) -> str:
    """
    메타데이터 입력용. 앞부분만 쓰면 '국회 의안 통합 요약'·목차만 보이고
    tl_summary가 그걸 베끼는 문제가 생긴다. 길면 앞·뒤를 나눠 [대상 텍스트]로 넘긴다.
    """
    s = (text or "").strip()
    if not s or len(s) <= max_chars:
        return s
    gap = len("\n\n...[중략]...\n\n")
    usable = max_chars - gap
    if usable < 200:
        return s[:max_chars]
    head = usable // 2
    tail = usable - head
    return s[:head] + "\n\n...[중략]...\n\n" + s[-tail:]


def build_big_categories_block(big_categories: list[dict] | None) -> str:
    """DB 대분류를 「대분류 후보 목록」+ `id: 이름` 나열 형태로 만든다."""
    if not big_categories:
        return ""
    pairs: list[str] = []
    for c in big_categories:
        cid = c.get("id")
        name = c.get("name", "") or ""
        if cid is None:
            continue
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            continue
        pairs.append(f"{cid_int}: {name}")
    if not pairs:
        return ""
    return "「대분류 후보 목록」 " + ", ".join(pairs) + "\n\n"


def metadata_user_prompt(
    summary: str,
    bc_block: str,
    *,
    max_summary_chars: int = 12000,
) -> str:
    """메타데이터 생성 요청 사용자 프롬프트. `max_summary_chars`는 스니펫 상한(앞·뒤 분할)."""
    snip = _metadata_snippet_from_summary(summary, max_summary_chars)
    bc_fallback = (
        ""
        if bc_block
        else "참고: 「대분류 후보 목록」이 없으면 둘째 줄은 'bc_id: 0'으로 출력한다.\n\n"
    )
    bc_pick_hint = (
        "지시 보강: 「대분류 후보 목록」이 있으면 bc_id는 그 목록에 나온 id 중 하나의 정수만, "
        "title 줄에는 어떤 꼬리표도 붙이지 않는다.\n\n"
        if bc_block
        else ""
    )
    return (
        f"{bc_block}"
        f"{bc_fallback}"
        f"{bc_pick_hint}"
        "지시: 아래 「대상 텍스트」만 읽고 title, bc_id, sc_keyword, tl_summary를 4줄로 작성한다. "
        "title은 비우지 말고, sc_keyword는 「대상 텍스트」에 실제 나온 단어만 쓴다. "
        "tl_summary는 title 문구를 반복하지 말고 원인-결과형 세 완결 문장(한 줄, 마침표 세 개 권장).\n\n"
        "「대상 텍스트」\n"
        f"{snip}\n"
    )
