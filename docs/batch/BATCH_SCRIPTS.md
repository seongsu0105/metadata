이 문서는 PDF 묶음에서 메타 4줄과 `finetune_dataset.jsonl`을 만드는 **`scripts/` 배치 파이프라인**의 파일·경로·환경 변수를 설명한다.

# 메타데이터 배치 스크립트 (`scripts/`)

PDF 묶음에서 **메타 4줄**(title, bc_id, sc_keyword, tl_summary)을 Ollama로 생성하고,  
`metadata.jsonl`과 (선택) `finetune_dataset.jsonl`을 쓰는 **오프라인 파이프라인**이다.  
프롬프트 정본은 형제 저장소 **`ollama-api`** 의 `app/prompts/summary.py` 를 import 한다.

---

## 파일 목록과 역할

| 파일 | 설명 |
|------|------|
| **`scripts/main.py`** | CLI 진입점. `batch_pipeline.process_all()` 만 호출한다. 실행: `cd scripts` 후 `python main.py`. |
| **`scripts/batch_config.py`** | 경로·환경 변수 헬퍼, `OLLAMA_API_ROOT` 로 워커 프롬프트 import, 대분류 `DB_BIG_CATEGORIES`, 출력 경로(`PDF_DIR`, `OUTPUT_DIR`, JSONL 경로). 모듈 docstring에 **파인튜닝 라벨 정의**(instruction / input / output 의미)가 적혀 있다. |
| **`scripts/batch_categories.py`** | `assign_bc_id`(키워드 점수), `normalize_bc_id`(모델 출력 교정 + 휴리스틱). |
| **`scripts/batch_parse.py`** | `parse_llm_metadata_block` — LLM 텍스트에서 `title:` 등 4키 파싱(전각 콜론·여러 줄 tl_summary 처리). |
| **`scripts/batch_postprocess.py`** | R1–R11 후처리 규칙, `MetadataPostprocessRules`, Kiwi 기반 키워드·요약 보조, `extract_title` / `extract_summary` / `format_metadata_block` 등. |
| **`scripts/batch_pdf.py`** | `clean_text`, `extract_pdf_text` — PyPDF2로 페이지 텍스트만 추출(스캔 PDF·복잡 레이아웃 한계는 주석으로 안내). |
| **`scripts/batch_llm.py`** | `dataset_user_prompt`(워커 `metadata_user_prompt` 래퍼), `generate_summary_ollama`(HTTP `/api/generate`, 재시도). |
| **`scripts/batch_record.py`** | `build_record_for_pdf` — Ollama 호출 → 파싱 → 후처리·폴백 → DB용 dict + 학습용 4줄 `output` 문자열. |
| **`scripts/batch_pipeline.py`** | `process_all` — `pdfdata` 스캔, (선택) 병렬, `out/` 에 JSONL 기록. PDF 폴더 없으면 `FileNotFoundError`. |

보조·레거시로 **`scripts/make_metadata.py`**, **`scripts/pure_metadata_export.py`** 등이 있을 수 있으나, 위 모듈이 **현재 권장 배치 경로**다.

---

## 디렉터리·산출물

| 경로 | 용도 |
|------|------|
| **`metadata/pdfdata/`** | 입력 PDF (기본). `METADATA_PDF_DIR` 로 변경 가능. |
| **`metadata/out/`** | 출력 디렉터리(자동 생성). `METADATA_OUTPUT_DIR` 로 변경 가능. |
| **`metadata/out/metadata.jsonl`** | DB·검수용 한 줄 JSON 레코드 (`source_pdf`, `title`, `bc_id`, `sc_keyword`, `tl_summary`). |
| **`metadata/out/finetune_dataset.jsonl`** | LoRA SFT용: `instruction` = `METADATA_SYSTEM`, `input` = Ollama에 보낸 user 프롬프트, `output` = **후처리 완료 4줄**(raw LLM 출력 아님). |
| **`scripts/metadata_sanitize_overrides.json`** | (선택) R11 치환 규칙 `from` / `to` 배열. |

---

## 환경 변수 요약

| 변수 | 의미 |
|------|------|
| `OLLAMA_API_ROOT` | `ollama-api` 저장소 루트 (미설정 시 `metadata` 형제 폴더 `ollama-api` 가정). |
| `METADATA_PDF_DIR` | PDF 입력 폴더. |
| `METADATA_OUTPUT_DIR` | JSONL 출력 폴더. |
| `METADATA_FAST` | `1` 이면 재시도·토큰·스니펫 상한 축소. |
| `METADATA_MAX_SUMMARY_CHARS` | `metadata_user_prompt` 스니펫 상한 (기본 12000 근처, FAST 시 상한 축소). |
| `METADATA_MAX_PDFS` | 처리할 PDF 개수 상한(0 또는 미설정 = 전부). |
| `METADATA_PARALLEL_WORKERS` | `ProcessPoolExecutor` 워커 수 (기본 1). |

---

## 의존성

- Python 패키지: `requests`, `PyPDF2`, `kiwipiepy` 등 (`metadata/requirements.txt` 참고).
- 런타임: 로컬 **Ollama** (`batch_llm.py` 의 `OLLAMA_URL` 기본값 `http://localhost:11434/api/generate`).
- 코드: 형제 **`ollama-api`** 의 `app.prompts.summary` (import 실패 시 안내 메시지와 함께 종료).

---

## 관련 문서

- 워커 프롬프트·계약: `ollama-api/app/prompts/summary.py`
- LoRA 학습 스크립트: `finetune/train_lora.py` (`finetune_dataset.jsonl` 소비)
- **Colab에서 학습할 때** 올릴 파일·순서: [`docs/finetune/colab/COLAB_TRAINING.md`](../finetune/colab/COLAB_TRAINING.md)
