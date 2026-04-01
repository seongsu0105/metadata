# Colab / Kaggle — LoRA 파인튜닝 (공통)

플랫폼별 **단계별 순서**는 아래 문서만 보면 된다.

- **[Colab 전용 실행 순서](README-colab.md)** — 노트북: `finetune/colab_train.ipynb`
- **[Kaggle 전용 실행 순서](README-kaggle.md)** — 노트북: `finetune/kaggle_train.ipynb`

`../scripts/main.py`로 만든 **`finetune_dataset.jsonl`**(각 줄: `instruction`, `input`, `output`)을 넣고, 이 폴더의 `train_lora.py`로 어댑터를 학습한다. 기본 베이스 모델은 **`MLP-KTLim/llama-3-Korean-Bllossom-8B`** 이다.

학습 직후 업로드가 필요하면 노트북의 `ENABLE_UPLOAD=True`와 `HF_REPO_ID`를 설정하고, `HF_TOKEN`(Colab/Kaggle Secrets)을 넣으면 노트북 셀 내부 `huggingface_hub` 호출로 자동 업로드된다.

## 로컬 vs Colab / Kaggle 역할 (꼭 확인)

- **노트북에는 `finetune_dataset.jsonl`을 꼭 올린다.** `train_lora.py`는 이 형식만 받는다. `metadata.jsonl`만으로는 이 스크립트를 그대로 쓸 수 없다.
- **`main.py`에서 `WRITE_FINETUNE = True`** 여야 `finetune_dataset.jsonl`이 생성된다. 파일이 없으면 로컬에서 `WRITE_FINETUNE`을 켜고 `main.py`를 다시 돌리거나, 동일 스키마로 직접 만든다.
- **`main.py`의 Ollama는 클라우드와 무관하다.** PDF 추출·Ollama HTTP 호출은 **로컬**(또는 Ollama가 떠 있는 PC)에서 끝낸 뒤, 노트북에는 **만들어진 JSONL + `train_lora.py` + `requirements-train.txt`**만 가져가면 된다.
- **게이트된 베이스 모델**을 쓰면 Hugging Face **`HF_TOKEN`**을 Colab Secrets 또는 Kaggle Secrets / 환경 변수로 넣는다.

## 준비물

- GPU 런타임(플랫폼별 선택은 위 링크 문서 참고)
- 데이터: `finetune_dataset.jsonl` 업로드 또는 드라이브·Kaggle Dataset 마운트
- (선택) 게이트 모델·비공개 허브 사용 시 **HF Access Token** (`HF_TOKEN`)

## 데이터 형식

한 줄 예시:

```json
{"instruction": "…", "input": "의안 OCR 텍스트 발췌…", "output": "title: …\nbc_id: …\nsc_keyword: …\ntl_summary: …"}
```

## 베이스 모델 바꾸기

`train_lora.py`의 `--base-model` 인자로 허브 id 또는 로컬 경로를 넘긴다. 기본값은 **`MLP-KTLim/llama-3-Korean-Bllossom-8B`** 이다.

| 후보 | 비고 |
|------|------|
| `MLP-KTLim/llama-3-Korean-Bllossom-8B` | 스크립트 기본값; Colab L4 또는 QLoRA(T4) 조합 권장 |
| `Qwen/Qwen2.5-1.5B-Instruct` | VRAM 매우 작을 때 테스트용 |
| `meta-llama/...` | 라이선스·토큰 필요할 수 있음 |

채팅 템플릿이 없는 모델은 스크립트가 `### 시스템` 형식으로 폴백한다.

## 자주 나는 오류

- **`CUDA out of memory`**: `--use-4bit`, `--batch-size 1`, `--max-length` 1024로 낮추기.
- **`chat_template` / 토크나이저 오류**: `transformers`·`trl` 버전을 `requirements-train.txt`에 맞추거나 최신으로 올리기.
- **SFTTrainer 인자 오류**: `trl` 메이저 버전마다 `tokenizer` vs `processing_class`가 다름. 스크립트가 둘 다 시도한다. 그래도 실패하면 `pip show trl` 버전을 알려 맞출 것.

## 로컬에서 돌릴 때

GPU 있는 Windows/WSL에서 동일 명령 가능. CPU만 있으면 매우 느리고 8B는 비현실적일 수 있다.

---

상위 단계(데이터 만드는 순서)는 `../메타데이터-작업-순서.md`를 본다.
