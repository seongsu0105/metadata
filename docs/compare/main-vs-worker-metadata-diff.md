이 문서는 로컬 배치 `scripts/main.py`와 ollama-api-worker 쪽 메타데이터·요약 파이프라인의 **함수 대응·차이·재현 한계**를 정리한다.

# `scripts/main.py` vs Worker (`metadata_source` + 요약 파이프라인) 차이

비교 대상:

- **배치(기준):** `scripts/main.py` — PDF → Ollama → 파싱 → 후처리까지 한 스크립트
- **Worker(백업):** `ollama-api-worker-changes-backup/app/vllm/summary/metadata_source.py`, `postprocess.py`, `pipeline.py`, `app/prompts/summary.py` 등

---

## 1. 구조

| 항목 | main.py | Worker |
|------|---------|--------|
| 역할 | 단일 스크립트 | 원문 보강·메타 후처리·추론이 모듈로 분리 |
| 메타 4줄 생성 | 스크립트 내 `METADATA_SYSTEM` + Ollama HTTP | vLLM + `summary.py` 프롬프트 + 파이프라인 |

---

## 2. `metadata_source.py`로 옮긴 것 / 대응

| main.py | Worker (`metadata_source` 등) |
|---------|--------------------------------|
| `extract_title` | `extract_title_from_source` (동일 휴리스틱: 앞 300자 `…법률안`, 없으면 앞 80자) |
| `reconcile_title_with_text` | `reconcile_title_for_metadata` — **`source_text` 없으면** 모델 title만 공백 정리 후 반환 (Worker 전용 분기) |
| `scrub_sc_keyword` | `scrub_sc_keyword_phrase` |
| `sc_keyword_plausible` + blob | `sc_keyword_parts_plausible` + `candidate_blob_for_keywords` (blob에 **요약·제목** 포함 가능) |
| `extract_keyword` (**Kiwi**) | `extract_keyword_candidates_from_title` (**Kiwi 없음**, 휴리스틱만) |
| `extract_summary` + 끊김 판정 일부 | `fallback_tl_summary_from_source` + `tl_summary_needs_source_fallback` (규칙 요약이 여전히 깨지면 `None`) |

---

## 3. main에만 있고 Worker 모듈에는 없거나 약한 것

1. **제목 정규화**  
   main: `reconcile_title_with_text`에서 `normalize_llm_output_text` (R1–R4, 치환 JSON R11 등).  
   Worker: 제목 쪽은 주로 `_normalize_title_ws` 수준; 나머지는 `postprocess`/다른 단계에 의존.

2. **요약 “오염 → 규칙 추출” 트리거 (R7 성격)**  
   main: `summary_should_use_extract_fallback`에 **베트남 문자**, **라틴/한글 비율** 등 포함.  
   `metadata_source`의 `tl_summary_needs_source_fallback`에는 **비율·베트남 검사 없음**.

3. **Kiwi**  
   main: `extract_keyword`, `loosen_dense_hangul_summary`(조밀 한글 요약 띄어쓰기).  
   Worker `metadata_source` 경로: Kiwi 없음.

4. **기타 main 전용**  
   `assign_bc_id` / `normalize_bc_id`, Ollama 옵션·재시도, `METADATA_QUALITY_RETRIES`, `parse_llm_metadata_block`, PDF I/O, `metadata_sanitize_overrides.json`, 파인튜닝 JSONL 등.

5. **프롬프트·모델**  
   main은 로컬 Ollama(`llama3.2:3b` 등); Worker는 vLLM + 별도 프롬프트. 문구가 한 글자라도 다르면 출력이 달라질 수 있음.

---

## 4. “그대로 실행하면 데이터셋이 전이랑 똑같이 나오나?”

**아니요. 동일하다고 보장할 수 없습니다.**

이유 요약:

- 모델·엔드포인트·프롬프트·샘플링 설정이 다를 수 있음.
- 제목/키워드/요약 후처리가 위 표대로 **1:1 동일하지 않음** (특히 Kiwi, R7류, `normalize_llm_output_text` 깊이).
- Worker는 `source_text` 부재 등 **입력 형태**가 배치 때와 다를 수 있음.
- 재현을 원하면 **같은 PDF 세트 + 같은 원문 추출 + 동일 프롬프트·온도·seed·모델**로 두 경로를 각각 돌려 **diff**(예: `title`/`tl_summary` 해시 또는 필드별 비교)를 측정하는 것이 안전합니다.

---

*작성 기준: 리포 내 `scripts/main.py` 및 `ollama-api-worker-changes-backup/` 백업 트리.*
