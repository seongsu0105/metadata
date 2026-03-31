# 메타데이터 사용처 정리

## 핵심

- `metadata.jsonl`, `finetune_dataset.jsonl`은 `app/prompts/summary.py`가 만드는 오프라인 산출물입니다.
- 이 Worker 레포 내부에서는 위 파일명을 직접 읽는 로직이 없습니다.
- Worker 런타임에서 실제로 쓰이는 건 `/api/vllm/process`에서 생성되는 `metadata_text`입니다.
- `metadata_text`는 SSE 응답으로 내려가고, 이후 저장/인덱싱(보통 메인 서버 또는 별도 파이프라인)에서 DB/Chroma 반영에 사용됩니다.

## 1) 오프라인 산출물의 의미

`app/prompts/summary.py` 주석 기준:

- `metadata.jsonl`: DB/배포용 메타데이터
- `finetune_dataset.jsonl`: (선택) LoRA 학습용 데이터셋

즉, 두 파일은 Worker API가 직접 읽는 입력이라기보다, 다른 단계로 넘기는 결과물 성격입니다.

## 2) Worker 내부 실사용 경로

### 2-1. 메타데이터 생성

`app/vllm/process_pipeline.py`에서:

1. 요약 생성
2. `metadata_user_prompt(...)` + `METADATA_SYSTEM`으로 메타 4줄 생성
3. `normalize_metadata_text(...)`, `repair_sc_keyword_from_summary(...)` 후처리
4. SSE 이벤트에 `metadata_text`로 포함

핵심 반환 예:

```python
yield wf._sse({"metadata_complete": True, "metadata_text": normalized_meta})
...
yield wf._sse(
    {
        "done": True,
        "summarize_text": summarize_text,
        "metadata_text": normalized_meta,
        "context_id": context_id,
    }
)
```

### 2-2. 라우터 전달

`app/routers/vllm.py`의 `POST /api/vllm/process`는 위 파이프라인 SSE를 그대로 스트리밍합니다.

## 3) RAG/Chroma에서의 메타 사용

- QA 검색 필터(`app/rag/documents/query.py`)는 주로 `access_level`, `owner_user_id`, `chunk_profile`를 사용합니다.
- 문맥 라벨(`app/rag/service/context_format.py`)에서는 `title`, `file_name`, `embedded_field` 등을 표시합니다.
- 메타 타입(`app/rag/documents/types.py`)에는 `bc_id`, `title`, `embedded_field` 등 저장 가능한 키가 정의되어 있습니다.

## 결론

- `metadata.jsonl` 자체는 Worker가 직접 소비하지 않습니다.
- Worker는 요청 시점에 메타를 생성해 `metadata_text`로 반환하고, 실제 저장/인덱싱 반영은 후속 시스템(메인 서버/별도 파이프라인) 책임입니다.

