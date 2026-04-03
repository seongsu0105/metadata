# =============================================================================
#   tests\test_summary_postprocess.py
# =============================================================================
"""`app.vllm.summary.postprocess` sanitize·구조화·예산 트림 (unittest)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.vllm.summary.postprocess import (
    build_structured_memo_text,
    finalize_metadata_line_output,
    repair_sc_keyword_from_summary,
    sanitize_partial_memo,
    sanitize_summary_output_text,
    strip_metadata_title_suffixes,
    trim_structured_memo_text_by_budget,
)


class TestSanitizePartialMemo(unittest.TestCase):
    def test_dedup_key(self) -> None:
        seen: dict[str, str] = {}
        t1 = sanitize_partial_memo("- 제목: A", seen)
        t2 = sanitize_partial_memo("- 제목: A", seen)
        self.assertIn("제목", t1)
        self.assertEqual(t2.count("제목"), 0)

    def test_empty_like_dropped(self) -> None:
        seen: dict[str, str] = {}
        out = sanitize_partial_memo("- 제목: 없음", seen)
        self.assertNotIn("제목: 없음", out)


class TestSanitizeSummaryOutput(unittest.TestCase):
    def test_strips_boilerplate_line(self) -> None:
        raw = (
            "아래는 구간에서 추출한 주요 사실을 bullet(-) 목록으로 정리한 내용입니다.\n"
            "실제 본문"
        )
        self.assertNotIn("아래는 구간에서", sanitize_summary_output_text(raw))


class TestBuildStructuredMemo(unittest.TestCase):
    def test_classify_sections(self) -> None:
        memos = [
            "- 의안번호: 123",
            "- 시행일: 2025-01-01",
        ]
        block = build_structured_memo_text(memos)
        self.assertIn("[식별정보]", block)
        self.assertIn("[시행·부칙]", block)


class TestMetadataTitleStrip(unittest.TestCase):
    def test_strip_bracket_bc_id(self) -> None:
        t = strip_metadata_title_suffixes("감염병 예방 관리에 관한 법률 일부 개정 법률안 요약 [bc_id]")
        self.assertEqual(t, "감염병 예방 관리에 관한 법률 일부 개정 법률안 요약")

    def test_finalize_rejects_bc_not_in_allowed(self) -> None:
        raw = (
            "title: 테스트\n"
            "bc_id: 5\n"
            "sc_keyword: a, b\n"
            "tl_summary: 한. 두. 세."
        )
        out = finalize_metadata_line_output(raw, allowed_bc_ids={1, 2, 3})
        self.assertIn("bc_id: 0", out)
        self.assertIn("title: 테스트", out)


class TestRepairMetadataWithSource(unittest.TestCase):
    def test_plausible_keyword_from_source_blob(self) -> None:
        source = "재생에너지 발전 설비 설치 촉진 특례법 일부개정법률안에 관하여 심의한다."
        summary = "## 제목\n일부 내용"
        norm = (
            "title: 요약\n"
            "bc_id: 0\n"
            "sc_keyword: 잘못된키, 없음키\n"
            "tl_summary: 첫째. 둘째."
        )
        out = repair_sc_keyword_from_summary(
            norm, summary, source_full_text=source
        )
        self.assertIn("재생에너지", out)
        self.assertIn("sc_keyword:", out)


class TestTrimByBudget(unittest.TestCase):
    def test_no_trim_when_under_budget(self) -> None:
        text = "[식별정보]\n- 제목: 테스트"
        out, trimmed, _tok = trim_structured_memo_text_by_budget(text, 100_000)
        self.assertFalse(trimmed)
        self.assertEqual(out.strip(), text.strip())
