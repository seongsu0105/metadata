# Kaggle 전용 — LoRA 파인튜닝 순서

공통 개요·데이터 스키마·자주 나는 오류는 [`README-colab-kaggle.md`](README-colab-kaggle.md)를 본다.
실행 노트북: `finetune/kaggle_train.ipynb`

## 런타임 GPU (권장)

- Notebook **Settings → Accelerator → GPU** 를 켠다. 배정되는 GPU는 **T4** 등 세션마다 다를 수 있다.
- **8B** 모델은 **OOM 방지용으로 `--use-4bit`** 를 기본으로 두는 것을 권장한다.

## 실행 순서

1. Notebook **Settings → Accelerator → GPU** 설정.
2. `finetune_dataset.jsonl`을 **Kaggle Dataset**으로 올리고 Notebook에 **Add Data**로 연결한다.
3. `train_lora.py`, `requirements-train.txt`는 **Code 업로드** 또는 Dataset에 같이 포함한다.
4. 데이터 경로는 보통 `/kaggle/input/<데이터셋-slug>/finetune_dataset.jsonl` 이다.
5. Hugging Face 토큰이 필요하면 **Add-ons → Secrets** 에 `HF_TOKEN` 을 넣고 노트북에서 환경 변수로 넘긴다.
6. 출력은 **`/kaggle/working/`** 아래만 세션 후에도 남는다(입력 데이터셋 경로에 쓰기 불가).

```bash
pip install -q -r /kaggle/input/<코드-데이터셋>/requirements-train.txt
python /kaggle/input/<코드-데이터셋>/train_lora.py \
  --data /kaggle/input/<데이터-데이터셋>/finetune_dataset.jsonl \
  --out /kaggle/working/lora-output \
  --base-model MLP-KTLim/llama-3-Korean-Bllossom-8B \
  --epochs 2 \
  --batch-size 2 \
  --grad-accum 4 \
  --max-length 2048 \
  --use-4bit
```

7. 자동 허브 업로드를 쓰려면:
   - Kaggle Secrets에 `HF_TOKEN` 저장
   - 노트북에서 `ENABLE_UPLOAD=True`, `HF_REPO_ID="<내 저장소>"` 설정
   - 학습 셀 완료 후 업로드 셀이 `huggingface_hub` API를 직접 호출해 `/kaggle/working/lora-output`을 업로드
8. 세션 종료 후 `/kaggle/working` 만 유지되므로, 업로드를 안 쓸 때는 어댑터를 zip으로 묶어 **Output**에서 내려받거나 다음 Dataset으로 올린다.
