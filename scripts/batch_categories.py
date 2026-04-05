from __future__ import annotations

import re
from typing import Any

from batch_config import VALID_BC_IDS


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
