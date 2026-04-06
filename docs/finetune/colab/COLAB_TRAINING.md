이 문서는 Google Colab(또는 동일 절차의 Kaggle)에서 **표준 `train_lora.py` 또는 Unsloth**로 LoRA를 학습할 때 준비할 파일·실행 순서를 안내한다.

# Google Colab에서 LoRA 학습하기

참고 문서: [학습 출력 형식·문체 5가지](../style/FINETUNE_STYLE_AND_FORMAT.md) · [LoRA 어댑터 옵션·적용 전후](../adapter/LORA_ADAPTER_GUIDE.md) · 리드용 요약: [ADAPTER_DEFINITION_FOR_LEAD.md](../adapter/ADAPTER_DEFINITION_FOR_LEAD.md) · **colab_train 실행→어댑터 5항목:** [COLAB_ADAPTER_RESULT_SUMMARY.md](COLAB_ADAPTER_RESULT_SUMMARY.md)

Colab에서는 아래 **두 가지 경로** 중 하나만 선택하면 됩니다. 데이터 형식은 동일합니다 (`finetune_dataset.jsonl`: `instruction` / `input` / `output`).

| 경로 | 노트북 | 학습 스크립트 | 특징 |
|------|--------|---------------|------|
| **표준 (HF + PEFT + TRL)** | `finetune/colab_train.ipynb` | `finetune/train_lora.py` | `requirements-train.txt`만으로 설치 가능. Unsloth 없음. |
| **선택 (Unsloth)** | `finetune/colab_unsloth_train.ipynb` | `finetune/train_lora_unsloth.py` | GPU 최적화·속도 이점 가능. unsloth git 설치 필요. |

---

## 1. Colab에 올릴 것(파일·구조)

리포지토리를 통째로 넣거나, 최소한 아래가 보이도록 맞춥니다.

```
/content/metadata/   (← 예시 루트, 노트북의 PROJECT_ROOT와 동일하게)
  finetune/
    train_lora.py
    train_lora_unsloth.py   ← Unsloth 경로만 쓸 때 필요
    requirements-train.txt
    colab_train.ipynb
    colab_unsloth_train.ipynb
  … (그 외 프로젝트 파일은 선택)
```

**학습 데이터** `finetune_dataset.jsonl`은 다음 중 편한 방법으로 준비합니다.

- Colab에 직접 업로드 → 노트북에서 `DATA_PATH = "/content/finetune_dataset.jsonl"` 등으로 지정
- Google Drive에 두고 마운트한 뒤 그 경로로 지정

노트북 기본값은 **`/content/finetune_dataset.jsonl`** 입니다 (`colab_train.ipynb` / `colab_unsloth_train.ipynb` 상단 변수에서 바꿀 수 있음).

게이트된 Hugging Face 모델을 쓰거나 어댑터를 허브에 올릴 때는 **런타임 비밀 또는 셀에서** `HF_TOKEN`(또는 `HUGGINGFACE_HUB_TOKEN`)을 설정합니다.

---

## 2. 표준 경로 실행 순서 (`colab_train.ipynb`)

1. **GPU 런타임** 연결: 런타임 → 런타임 유형 변경 → GPU.
2. **프로젝트 배치**: `metadata`를 `/content/metadata` 등에 두고, 노트북 맨 위 **`PROJECT_ROOT`**, **`DATA_PATH`**, **`OUTPUT_DIR`**, **`BASE_MODEL`** 확인·수정.
3. **패키지 설치** 셀 실행: `requirements-train.txt` + `huggingface_hub` 등.
4. **학습 실행** 셀: `train_lora.py`를 `subprocess`로 호출 (노트북에 이미 정의됨).
5. **결과**: `OUTPUT_DIR`(기본 `/content/lora-output`)에 어댑터·토크나이저 저장. `ENABLE_UPLOAD=True`면 허브 업로드 셀 추가 실행(노트북 후반에 있으면 그대로 따름).

---

## 3. Unsloth 경로 실행 순서 (`colab_unsloth_train.ipynb`)

1. **GPU 런타임** 연결.
2. **프로젝트·데이터 경로** 설정 (`PROJECT_ROOT`, `DATA_PATH`, `OUTPUT_DIR` 등) — 표준 경로와 동일한 개념.
3. **Unsloth 설치** 셀 실행  
   - **권장**: 노트북 **[A] 셀** — `unsloth[colab-new]` git 설치 (의존성 포함).  
   - **충돌 시**: **[B] 셀** 주석 해제 — 팀에서 쓰는 `uninstall` + `--no-deps` 레시피.
4. **학습 실행** 셀: `train_lora_unsloth.py` 호출.
5. **결과**: `OUTPUT_DIR`(기본 `/content/lora-output-unsloth`)에 저장. W&B·허브는 `REPORT_TO`, `PUSH_TO_HUB`, `HUB_MODEL_ID`, `HF_TOKEN`으로 노트북 상단에서 켬.

Unsloth는 **Linux GPU(Colab)** 위주 지원입니다. Windows 로컬에서는 동일 스크립트가 동작하지 않을 수 있습니다.

---

## 4. 로컬에서 `finetune_dataset.jsonl` 만들기 (참고)

PDF 메타데이터 배치는 `scripts/main.py` 등으로 `finetune_dataset.jsonl`을 생성할 수 있습니다. Colab에 **이 jsonl만** 복사해 올려도 학습은 가능합니다 (프로젝트 전체는 선택).

---

## 5. 정리

- **무엇을 돌리나**: 표준은 `colab_train.ipynb` → `train_lora.py` / Unsloth는 `colab_unsloth_train.ipynb` → `train_lora_unsloth.py`.
- **반드시 맞출 것**: `DATA_PATH`의 jsonl, `PROJECT_ROOT` 아래 `finetune/` 스크립트 존재, GPU 런타임.
- **선택**: `HF_TOKEN`, W&B, 허브 업로드 (`colab_train`은 기존 업로드 셀·변수, Unsloth 노트북은 상단 플래그).
