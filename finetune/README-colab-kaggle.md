# Colab / Kaggle — LoRA 파인튜닝

`../scripts/main.py`로 만든 **`finetune_dataset.jsonl`**(각 줄: `instruction`, `input`, `output`)을 넣고, 이 폴더의 `train_lora.py`로 어댑터를 학습한다.

## 로컬 vs Colab 역할 (꼭 확인)

- **Colab에는 `finetune_dataset.jsonl`을 꼭 올린다.** `train_lora.py`는 이 형식(`instruction` / `input` / `output`)만 받는다. `metadata.jsonl`만으로는 이 스크립트를 그대로 쓸 수 없다.
- **`main.py`에서 `WRITE_FINETUNE = True`** 여야 `finetune_dataset.jsonl`이 생성된다. 파일이 없으면 로컬에서 `WRITE_FINETUNE`을 켜고 `main.py`를 다시 돌리거나, 동일 스키마로 직접 만든다.
- **`main.py`의 Ollama는 Colab과 무관하다.** PDF 추출·Ollama HTTP 호출은 **로컬**(또는 Ollama가 떠 있는 PC)에서 끝낸 뒤, Colab에는 **만들어진 JSONL + `train_lora.py` + `requirements-train.txt`**(또는 동일 pip 목록)만 가져가면 된다.
- **Colab:** 런타임을 **GPU**로 바꾸고, `colab_kaggle.ipynb`의 **`DATA_PATH` / `OUTPUT_DIR`**(또는 터미널의 `--data` / `--out`)을 본인 업로드 경로에 맞게 수정한다.
- **게이트된 베이스 모델**(일부 Llama 등)을 쓰면 Hugging Face **`HF_TOKEN`**을 Colab Secrets 또는 환경 변수로 넣는다.

## 준비물

- GPU 런타임: Colab **GPU**(T4 이상 권장) / Kaggle Notebook **GPU** 켜기
- 데이터: `finetune_dataset.jsonl` 업로드 또는 드라이브·Kaggle Dataset으로 마운트
- (선택) 게이트 모델 사용 시 Hugging Face **Access Token** (`HF_TOKEN`)

## 데이터 형식

한 줄 예시:

```json
{"instruction": "…", "input": "의안 OCR 텍스트 발췌…", "output": "title: …\nbc_id: …\nsc_keyword: …\ntl_summary: …"}
```

## Colab에서 실행 순서

1. **런타임 → 런타임 유형 변경 → GPU**
2. 이 폴더를 업로드하거나 Git으로 가져오기 (`train_lora.py`, `requirements-train.txt`).
3. `finetune_dataset.jsonl`을 `/content/` 등에 업로드.
4. 셀에서 설치:

```bash
pip install -q -r requirements-train.txt
```

5. 학습 실행:

```bash
python train_lora.py \
  --data /content/finetune_dataset.jsonl \
  --out /content/drive/MyDrive/lora-metadata \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --epochs 2 \
  --batch-size 2 \
  --grad-accum 4 \
  --max-length 2048
```

6. VRAM이 부족하면 `--use-4bit` 추가.

7. 결과물 `--out` 폴더: LoRA 가중치 + `tokenizer` 설정. 추론 시 베이스 모델 id와 함께 `PeftModel.from_pretrained`로 로드.

## Kaggle에서 실행 순서

1. Notebook **Settings → Accelerator → GPU** 선택.
2. `finetune_dataset.jsonl`을 **Kaggle Dataset**으로 올리고 Notebook에 **Add Data**로 연결.
3. `train_lora.py` 등을 **Code 탭에 업로드**하거나 Dataset에 같이 넣는다.
4. 데이터 경로는 보통 `/kaggle/input/<dataset-name>/finetune_dataset.jsonl`.
5. 출력은 **`/kaggle/working/`** 아래에 저장 (제출·다운로드 가능).

```bash
pip install -q -r /kaggle/input/<your-code-dataset>/requirements-train.txt
python /kaggle/input/<your-code-dataset>/train_lora.py \
  --data /kaggle/input/<your-data-dataset>/finetune_dataset.jsonl \
  --out /kaggle/working/lora-output \
  --epochs 2 \
  --use-4bit
```

6. 세션 끝나면 `/kaggle/working`만 유지되므로, 어댑터는 zip으로 묶어 **Output**에 내려받거나 다음 Dataset으로 올린다.

## 베이스 모델 바꾸기

| 후보 | 비고 |
|------|------|
| `Qwen/Qwen2.5-1.5B-Instruct` | 기본값, Colab T4에 무난 |
| `Qwen/Qwen2.5-3B-Instruct` | VRAM 여유 있을 때 |
| `meta-llama/Llama-3.2-3B-Instruct` | HF에서 라이선스 동의 + 토큰 필요할 수 있음 |
| EXAONE 등 | 허브에 Instruct 버전이 있으면 동일 방식으로 `--base-model`만 변경 |

채팅 템플릿이 없는 모델은 스크립트가 `### 시스템` 형식으로 폴백한다.

## 자주 나는 오류

- **`CUDA out of memory`**: `--use-4bit`, `--batch-size 1`, `--max-length` 1024로 낮추기.
- **`chat_template` / 토크나이저 오류**: `transformers`·`trl` 버전을 `requirements-train.txt`에 맞추거나 최신으로 올리기.
- **SFTTrainer 인자 오류**: `trl` 메이저 버전마다 `tokenizer` vs `processing_class`가 다름. 스크립트가 둘 다 시도한다. 그래도 실패하면 `pip show trl` 버전을 알려 팀에서 맞출 것.

## 로컬에서 돌릴 때

GPU 있는 Windows/WSL에서 동일 명령 가능. CPU만 있으면 매우 느리고 일부 모델은 비현실적이다.

---

상위 단계(데이터 만드는 순서)는 `../메타데이터-작업-순서.md`를 본다.
