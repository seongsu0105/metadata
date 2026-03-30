# Colab 전용 — LoRA 파인튜닝 순서

공통 개요·데이터 스키마·자주 나는 오류는 [`README-colab-kaggle.md`](README-colab-kaggle.md)를 본다.

## 런타임 GPU (권장)

- **우선:** **L4 GPU** — 8B + LoRA에 VRAM 여유가 있어 `--use-4bit` 없이도 시도해 볼 수 있다(OOM 나면 4bit).
- **무료/제한:** **T4 GPU** — 8B는 **반드시 `--use-4bit`** 권장, 필요 시 `--batch-size 1`, `--max-length` 축소.

(Colab 메뉴: **런타임 → 런타임 유형 변경 → 하드웨어 가속기 → GPU**에서 위 GPU가 보이면 그중 **L4**를 고른다.)

## 실행 순서

1. **런타임 → 런타임 유형 변경 → GPU** (위 권장 GPU 선택).
2. 이 폴더 업로드 또는 Git 클론 (`train_lora.py`, `requirements-train.txt`).
3. `finetune_dataset.jsonl`을 `/content/` 또는 Drive에 둔다.
4. Hugging Face에서 모델·토큰 제한이 있으면 **Secrets**에 `HF_TOKEN` 저장 후 노트북에서 로드.

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")  # Colab Secrets 이름과 동일하게
```

5. 의존성 설치:

```bash
pip install -q -r requirements-train.txt
```

6. 학습 (베이스: `MLP-KTLim/llama-3-Korean-Bllossom-8B`, 출력은 Drive 등 본인 경로로 변경):

```bash
python train_lora.py \
  --data /content/finetune_dataset.jsonl \
  --out /content/drive/MyDrive/lora-metadata \
  --base-model MLP-KTLim/llama-3-Korean-Bllossom-8B \
  --epochs 2 \
  --batch-size 2 \
  --grad-accum 4 \
  --max-length 2048
```

7. **T4**이거나 OOM이면 `--use-4bit`를 명령 끝에 추가한다.

8. `--out` 폴더: LoRA 가중치 + 토크나이저. 추론 시 동일 베이스 id와 `PeftModel.from_pretrained`로 로드한다.
