# =============================================================================
#  app\vllm\summary\pipeline.py
# =============================================================================
"""
`POST /api/vllm/process` — OCR 등 받아 부분청킹→최종요약(SSE).
`docs/ex/api-files-save-summarize-text.md` 의 이벤트 순서만 클라이언트에 노출한다.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from app.prompts.summary import (
    METADATA_SYSTEM,
    SUMMARY_SYSTEM_FINAL,
    build_big_categories_block,
    metadata_user_prompt,
    summary_user_single_document,
)
from app.vllm import workflow as wf
from app.vllm.summary.korean_token_optimizer import (
    maybe_optimize_for_summary_llm,
    summary_morph_optimize_enabled,
)
from app.vllm.summary.pipeline_helpers import (
    SUMMARY_RAG_BODY,
    log_summary_llm_call,
    user_option_suffix,
)
from app.vllm.summary.pipeline_map_reduce import stream_map_reduce_branch
from app.vllm.summary.chunking import (
    chunk_size_and_overlap_for_pages,
    chunk_text_with_page_metadata,
    estimate_page_count,
)
from app.vllm.summary.metrics import (
    approx_tokens_for_text,
    log_chunking_overview,
    log_morph_single,
    summary_context_limit_tokens,
)
from app.vllm.summary.postprocess import (
    allowed_bc_ids_from_payload,
    finalize_metadata_line_output,
    normalize_metadata_text,
    repair_sc_keyword_from_summary,
    sanitize_summary_output_text,
)
from app.vllm.summary.process_memo_rag_ingest import ingest_process_memos_for_qa_sync

_log = logging.getLogger(__name__)


async def run_llm_process_pipeline_sse(
    *,
    full_text: str,
    ocr_result: dict | None,
    context_id: str,
    access_level: str,
    user_id: int,
    big_categories: list[dict] | None = None,
    metadata_max_summary_chars: int = 12000,
    metadata_options: dict[str, object] | None = None,
    rag_ingest: dict[str, object],
) -> AsyncIterator[str]:
    """
    단문이면 6단원 직접 스트리밍, 장문이면 부분 bullet → 메모 합성 6단원 스트리밍.
    SSE: summarizing → summary_delta* → summary_complete → done (메타 포함, 문서와 동일).
    메타 LLM 직후 항상 ``rag_ingest`` 기준으로 QA 인제스트(임베딩·Chroma)를 수행한다.
    """
    try:
        model = wf._resolve_chat_model(body_model=None)
    except ValueError as e:
        _log.warning("요약 파이프라인 모델 해석 실패 user_id=%s: %s", user_id, e)
        yield wf._sse({"error": str(e)})
        return

    morph_on = summary_morph_optimize_enabled()
    max_ctx = summary_context_limit_tokens()
    _log.info(
        "요약 파이프라인 시작 user_id=%s model=%s text_len=%d morph_optimize=%s max_ctx=%s",
        user_id,
        model,
        len(full_text),
        morph_on,
        max_ctx,
    )

    est_pages = estimate_page_count(ocr_result, full_text)
    csize, ov = chunk_size_and_overlap_for_pages(est_pages)
    chunks_meta = chunk_text_with_page_metadata(
        full_text=full_text,
        ocr_result=ocr_result,
        chunk_size=csize,
        overlap=ov,
    )
    if not chunks_meta:
        _log.warning("요약 파이프라인 청크 없음 user_id=%s", user_id)
        yield wf._sse({"error": "청크 분할 결과가 비어 있습니다."})
        return

    ft_toks, ft_m = approx_tokens_for_text(full_text)
    log_chunking_overview(
        full_text_chars=len(full_text),
        full_text_approx_tokens=ft_toks,
        full_text_method=ft_m,
        est_pages=est_pages,
        chunk_count=len(chunks_meta),
        chunk_size=csize,
        overlap=ov,
        max_ctx=max_ctx,
    )

    opt_suffix = user_option_suffix(access_level=access_level, user_id=user_id)

    single_shot = len(chunks_meta) == 1
    yield wf._sse(
        {
            "pipeline": {
                "mode": "single" if single_shot else "map_reduce",
                "estimated_pages": est_pages,
                "chunk_size": csize,
                "overlap": ov,
                "chunk_count": len(chunks_meta),
                "morph_optimize": morph_on,
            }
        }
    )
    _log.info(
        "요약 파이프라인 청킹 mode=%s est_pages=%s chunk_count=%d csize=%d overlap=%d",
        "single" if single_shot else "map_reduce",
        est_pages,
        len(chunks_meta),
        csize,
        ov,
    )

    user_final = ""
    partial_memos_holder: list[tuple[str, str, int]] = []

    if single_shot:
        body_for_llm = maybe_optimize_for_summary_llm(full_text)
        if morph_on:
            tb, tbm = approx_tokens_for_text(full_text)
            ta, tam = approx_tokens_for_text(body_for_llm)
            log_morph_single(
                chars_before=len(full_text),
                chars_after=len(body_for_llm),
                tok_before=tb,
                tok_after=ta,
                method_before=tbm,
                method_after=tam,
            )
        user_final = summary_user_single_document(body_for_llm) + opt_suffix
    else:
        user_final_holder: list[str] = []
        async for line in stream_map_reduce_branch(
            model=model,
            user_id=user_id,
            chunks_meta=chunks_meta,
            opt_suffix=opt_suffix,
            morph_on=morph_on,
            max_ctx=max_ctx,
            user_final_holder=user_final_holder,
            partial_memos_holder=partial_memos_holder,
        ):
            yield line
        if not user_final_holder:
            return
        user_final = user_final_holder[0]

    yield wf._sse({"summarizing": True})
    _log.info("요약 최종 스트림 시작 mode=%s", "single" if single_shot else "map_reduce")
    parts2: list[str] = []
    _merged_final = wf._merge_prompt_options(None, body=SUMMARY_RAG_BODY)
    _samp_final = wf._normalize_sampling(_merged_final)
    final_max_tokens = _samp_final.get("max_tokens")
    usage_final: list = []
    try:
        async for delta in wf._generate_stream(
            model,
            SUMMARY_SYSTEM_FINAL,
            user_final,
            options=None,
            body=SUMMARY_RAG_BODY,
            usage_out=usage_final,
        ):
            parts2.append(delta)
            yield wf._sse({"summary_delta": delta})
    except Exception as e:
        _log.exception("요약 map-reduce 최종 스트림 실패 user_id=%s", user_id)
        yield wf._sse({"error": str(e)})
        return

    summarize_text = sanitize_summary_output_text("".join(parts2))
    log_summary_llm_call(
        phase="final",
        user_id=user_id,
        usage_out=usage_final,
        system=SUMMARY_SYSTEM_FINAL,
        prompt=user_final,
        output_text="".join(parts2),
        max_tokens_request=final_max_tokens if isinstance(final_max_tokens, int) else None,
        max_ctx=max_ctx,
    )
    yield wf._sse({"summary_complete": True})
    _log.info(
        "요약 최종 스트림 완료 mode=%s summary_chars=%d",
        "single" if single_shot else "map_reduce",
        len(summarize_text),
    )

    done_evt: dict[str, object] = {
        "done": True,
        "summarize_text": summarize_text,
        "context_id": context_id,
    }
    md_map: dict[str, str] = {
        "title": "원문상 명시 없음",
        "bc_id": "0",
        "sc_keyword": "원문상 명시 없음",
        "tl_summary": "원문상 명시 없음.",
    }
    try:
        bc_block = build_big_categories_block(big_categories)
        md_user = metadata_user_prompt(
            summarize_text,
            bc_block=bc_block,
            max_summary_chars=metadata_max_summary_chars,
        )
        md_raw = await wf.generate_metadata_llm_text(
            system=METADATA_SYSTEM,
            user=md_user,
            options=metadata_options,
        )
        md_norm = normalize_metadata_text(md_raw)
        md_fixed = repair_sc_keyword_from_summary(
            md_norm,
            summarize_text,
            source_full_text=full_text,
        )
        md_fixed = finalize_metadata_line_output(
            md_fixed,
            allowed_bc_ids=allowed_bc_ids_from_payload(big_categories),
        )
        md_map = {}
        for ln in md_fixed.splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                md_map[k.strip()] = v.strip()
        done_evt["metadata"] = md_map
        done_evt["metadata_text"] = md_fixed
        done_evt["metadata_raw"] = md_raw
    except Exception:
        _log.exception("요약 메타데이터 생성 실패 user_id=%s", user_id)
        done_evt["metadata"] = md_map
        done_evt["metadata_text"] = ""
        done_evt["metadata_raw"] = ""

    try:
        bc_meta = int(str(md_map.get("bc_id") or "0").strip() or "0")
    except ValueError:
        bc_meta = 0
    n_chunks, ing_err = await asyncio.to_thread(
        ingest_process_memos_for_qa_sync,
        full_text=full_text,
        access_level=access_level,
        user_id=user_id,
        bc_id=bc_meta,
        rag=rag_ingest,
        ocr_result=ocr_result,
        partial_memos=partial_memos_holder,
        final_summary=summarize_text,
    )
    done_evt["rag_ingest_chunks"] = n_chunks
    if ing_err:
        done_evt["rag_ingest_error"] = ing_err
    fid_raw = str(rag_ingest.get("file_id") or "").strip()
    _log.info(
        "요약 후 QA 임베딩/Chroma 인제스트 결과 file_id=%s memos=%d error=%s",
        fid_raw,
        n_chunks,
        ing_err or "-",
    )

    yield wf._sse(done_evt)
    _log.info("요약 파이프라인 done 이벤트 전송 summarize_chars=%d", len(summarize_text))
