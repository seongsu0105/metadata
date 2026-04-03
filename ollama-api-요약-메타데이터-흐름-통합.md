# ollama-api — 요약·메타데이터 흐름 통합 정리

> 생성일: 2026-04-02  
> 원본 저장소: `C:\Users\USER\Desktop\ollama-api`  
> 이 문서는 **대화에서 정리한 파일 맵**, 저장소 **`docs/summary/*.md` 요지**, 그리고 **실제 코드 기준 보충**을 한 파일로 묶은 것입니다.

---

## 1. 한눈에 보는 두 갈래

| 구분 | 무엇 | 벡터 검색 |
|------|------|-----------|
| **의안 OCR 요약 (HTTP)** | `POST /api/vllm/process` — OCR `pages` 합본 → 문자 청킹 → single 또는 map-reduce → 6단원 요약 → 메타 4줄 LLM → (선택) QA 인제스트 | 요약 경로에서 **유사도 검색 없음** |
| **file_id 순서 요약 RAG (비 HTTP)** | 이미 Chroma에 `summary` 프로필로 인제스트된 청크를 **페이지/인덱스 순**으로 모아 stride 샘플링 후 문맥 블록 생성 | **유사도 검색 없음** (순서 스캔) |

---

## 2. HTTP 진입 ~ 합본 원문

| 파일 | 역할 |
|------|------|
| `app/main.py` | `app.include_router(vllm_router, prefix="/api/vllm")` → 실제 URL은 **`POST /api/vllm/process`** |
| `app/routers/vllm.py` | JWT(`WORKER_JWT_SECRET`) 검증, `extract_summary_input_text`, `run_llm_process_pipeline_sse`로 **SSE** 스트림 |
| `app/vllm/process/request.py` | `ocr_result.pages`를 페이지 키 순으로 `"\n\n"` 이은 **합본 문자열** (`extract_summary_input_text`, `ocr_pages_merged_text_and_spans`) |
| `app/schemas/llm_process.py` | `LLMProcessRequest` — `file_id`(필수), `access_level`, `user_id`, `big_categories`, `metadata_max_summary_chars`, `metadata_options`, `rag_ingest` 및 병합(`rag_ingest_merged`) |
| `app/vllm/process/auth.py` | `verify_process_bearer_or_raise` — 시크릿 없으면 로컬에서 검증 생략 |

**참고 계약:** 코드·주석에서 인용하는 `docs/ex/api-files-save-summarize-text.md`는 저장소에 없을 수 있음 — 있으면 SSE 이벤트·필드 정의의 기준 문서로 보면 됨.

---

## 3. 요약 파이프라인 본체

**중심 파일:** `app/vllm/summary/pipeline.py` — `run_llm_process_pipeline_sse`

### 3.1 실행 순서 (요약)

1. `app/vllm/workflow.py` — `_resolve_chat_model`로 모델 확정  
2. `app/vllm/summary/chunking.py` — `estimate_page_count` → `chunk_size_and_overlap_for_pages` → `chunk_text_with_page_metadata`  
3. SSE **`pipeline`** 이벤트: `mode`(single | map_reduce), `estimated_pages`, `chunk_size`, `overlap`, `chunk_count`, `morph_optimize`  
4. **단발:** `app/prompts/summary.py`의 `summary_user_single_document` + 형태소 최적화 옵션  
5. **장문:** `app/vllm/summary/pipeline_map_reduce.py` — 구간별 `SUMMARY_SYSTEM_PARTIAL` / `summary_user_partial` 스트리밍, 메모 합성·예산·2차 압축 후 `summary_user_final_from_chunk_memos`  
6. **`summarizing`** → `app/vllm/workflow.py`의 `_generate_stream` — 시스템 `SUMMARY_SYSTEM_FINAL`, **`body`에 요약 모드**(예: `SUMMARY_RAG_BODY` / `rag.mode: summary`)로 **요약 전용 샘플링 기본값** 병합  
7. `app/vllm/summary/postprocess.py` — `sanitize_summary_output_text` 등  
8. SSE: `summary_delta*` → `summary_complete`

**용어 (온보딩용):**

- **합본 원문:** OCR `pages`만으로 만든 문자열 (`extract_summary_input_text`).  
- **청킹 좌표(provenance):** 합본 기준 문자 구간·가능하면 PDF 페이지 라벨 — `summary_user_partial`에 포함.  
- **`rag.mode: summary` in body:** Chroma 검색을 켜는 플래그가 **아니라**, `workflow`에서 **요약용 샘플링 기본값**을 고르는 용도.

### 3.2 관련 파일 목록 (요약 전용)

| 파일 | 내용 |
|------|------|
| `app/prompts/summary.py` | `SUMMARY_SYSTEM_FINAL` / `PARTIAL`, 유저 프롬프트 조립, **`METADATA_SYSTEM`**, `build_big_categories_block`, `metadata_user_prompt` |
| `app/vllm/summary/pipeline_helpers.py` | `SUMMARY_RAG_BODY`, 로깅, `user_option_suffix` 등 |
| `app/vllm/summary/korean_token_optimizer.py` | 형태소 최적화 선택 |
| `app/vllm/summary/metrics.py` | 토큰 추정, 컨텍스트 한도 로깅 |
| `app/config.py` | `get_summary_default_prompt_options`, `SUM_*` 등 환경 연동 |

---

## 4. 메타데이터 (title / bc_id / sc_keyword / tl_summary)

**위치:** `app/vllm/summary/pipeline.py` — 최종 `summarize_text` 확정 **이후**, 마지막 `done` SSE **직전**.

| 단계 | 파일 | 설명 |
|------|------|------|
| 프롬프트 | `app/prompts/summary.py` | `METADATA_SYSTEM`(4줄 고정), `build_big_categories_block`, `metadata_user_prompt`(요약 스니펫 + 대분류 후보) |
| LLM | `app/vllm/workflow.py` | **`generate_metadata_llm_text`** — 비스트림, `get_metadata_default_prompt_options`, 선택 `VLLM_METADATA_LORA` |
| 후처리 | `app/vllm/summary/postprocess.py` | `normalize_metadata_text`, `repair_sc_keyword_from_summary`, `finalize_metadata_line_output`, `allowed_bc_ids_from_payload` |
| 응답 | `pipeline.py` | `done` 이벤트에 `metadata`(dict), `metadata_text`, `metadata_raw` — 실패 시 placeholder dict |

> **문서 주의:** `docs/summary/summary-code-guide.md` §7에 “메타데이터 전용 LLM을 하지 않는다”는 문장이 있으나, **현재 코드는 위와 같이 호출함.** 판단은 `pipeline.py` 기준.

---

## 5. ` done` 이후 — QA용 Chroma 인제스트 (벡터 메타)

| 파일 | 역할 |
|------|------|
| `app/vllm/summary/process_memo_rag_ingest.py` | `ingest_process_memos_for_qa_sync` — 부분 메모·최종 요약을 Chroma에 삽입 (`content_type` 등) |
| `app/vllm/summary/process_rag_ingest.py` | `rag_ingest` 페이로드를 벡터 메타 베이스로 정리 |
| `app/rag/documents/types.py` | `FileVectorMetadata`, `file_vector_metadata_from_file_row` — `file_id`, `owner_user_id`, `access_level`, `title`, `embedded_field`, `chunk_profile` 등 |
| `app/rag/documents/ingest.py` | 청크 삽입, `coerce_chroma_metadata` |

**구분:** 클라이언트로 나가는 **`metadata`(4줄 LLM 출력)** 과, Chroma 문서에 붙는 **필터/검색용 메타**는 목적과 스키마가 다름.

`done` 이벤트에는 `rag_ingest_chunks`, 실패 시 `rag_ingest_error` 등이 포함될 수 있음.

---

## 6. file_id 기반 “요약 RAG” (HTTP 아님)

저장소 문서 `docs/summary/worker-summary-rag-flow.md` 요지:

```text
run_single_generate(payload)
  → app/vllm/workflow.py
       → app/rag/service/context.py          build_rag_context_block
            → app/rag/service/rag_payload.py    normalize_rag_payload (mode = summary)
            → app/rag/chroma/ordered_chunks.py  fetch_chunks_ordered_by_file
            → app/rag/config/summary_chunk_presets
            → app/rag/modes/summary/chunk_stride_selection.py
            → app/rag/service/context_format.py format_context_block
       → inject_rag_into_messages → vLLM
```

**사전 조건:** 해당 `file_id`로 **`summary` 프로필** 청크가 Chroma에 있어야 함.  
원문 청킹·저장 시점: `app/rag/documents/ingest.py` (`ingest_long_text_as_chunks` 등).

---

## 7. SSE 이벤트 순서 (참고)

- **단발:** `pipeline` → `summarizing` → `summary_delta*` → `summary_complete` → (메타·인제스트) → `done`  
- **장문:** `pipeline` → 각 구간 `partial_*` → (필요 시 2차 축약) → `partial_memos_combined` → `summarizing` → `summary_delta*` → `summary_complete` → `done`

오류 시 중간에 `error` 이벤트.

---

## 8. 팀 온보딩 — 읽기 순서 (저장소 `summary-code-guide` §13과 동일 취지)

1. 이 통합 문서 §0~2 (맥락)  
2. `docs/ex/api-files-save-summarize-text.md` (있을 경우 계약)  
3. `app/routers/vllm.py`  
4. `app/schemas/llm_process.py`  
5. `app/vllm/process/request.py`  
6. **`app/vllm/summary/pipeline.py`**  
7. `app/vllm/summary/pipeline_map_reduce.py`  
8. `app/vllm/summary/chunking.py`  
9. `app/prompts/summary.py`  
10. `app/vllm/workflow.py` (`_sse`, `_generate_stream`, **`generate_metadata_llm_text`**)  
11. `app/vllm/summary/postprocess.py`  
12. `app/vllm/process/auth.py`  
13. `app/config.py`  

**QA·벡터 검색(질의):** `docs/qa/rag-qa-code-guide.md`

---

## 9. 저장소 내 원문 MD 경로 (복사 없이 참조용)

| 경로 | 내용 |
|------|------|
| `ollama-api/docs/summary/summary-code-guide.md` | SSE·파이프라인 상세 온보딩 (일부 문장은 코드와 불일치 가능 — §4 참고) |
| `ollama-api/docs/summary/worker-summary-rag-flow.md` | Chroma 순서 요약 RAG 파일 맵 |
| `ollama-api/docs/summary/summary-strategy-catalog.md` | 전략·진입점 표 형태 정리 |
| `ollama-api/docs/qa/qa-rag-by-file-id-guide.md` | `file_id` 인제스트·QA 연동 |

---

*이 파일은 데스크톱 `C:\Users\USER\Desktop\ollama-api-요약-메타데이터-흐름-통합.md` 에 저장되어 있습니다.*
