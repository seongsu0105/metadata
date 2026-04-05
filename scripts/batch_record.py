from __future__ import annotations

from typing import Any

from batch_categories import normalize_bc_id
from batch_config import effective_quality_retries
from batch_llm import dataset_user_prompt, generate_summary_ollama
from batch_parse import parse_llm_metadata_block
from batch_postprocess import (
    RULES,
    extract_keyword,
    extract_summary,
    format_metadata_block,
    is_tl_summary_truncated_or_broken,
    loosen_dense_hangul_summary,
    normalize_llm_output_text,
    reconcile_title_with_text,
    scrub_sc_keyword,
    sc_keyword_plausible,
    summary_is_acceptable,
    summary_quality_score,
    summary_should_use_extract_fallback,
    _strip_long_latin_tokens,
)


def build_record_for_pdf(
    source_pdf: str, text: str, bc_block: str
) -> tuple[dict[str, Any], str]:
    """
    한 PDF(또는 문서) 텍스트에 대해 DB용 dict와 학습용 output 블록(후처리 완료 4줄)을 생성.
    """
    best_meta_row: dict[str, Any] | None = None
    best_block = ""
    best_score = -1

    for _ in range(effective_quality_retries() + 1):
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

    if best_meta_row is not None:
        return best_meta_row, best_block
    raise RuntimeError("메타데이터 생성 실패: 유효한 요약을 만들지 못했습니다.")
