이 문서는 이 저장소에서 쓰는 **PEFT LoRA 어댑터**의 종류, `adapter_config.json` 필드, CLI 차이, 적용 전후, 흔한 실수를 설명한다.

# LoRA 어댑터 가이드 — 종류·옵션·성격·적용 전후

이 저장소에서 말하는 “어댑터”는 **Hugging Face PEFT의 LoRA(Low-Rank Adaptation)** 입니다. 베이스 LM 가중치는 고정하고, 일부 선형층에 **작은 rank 행렬**만 추가로 학습합니다.

관련: [출력 형식·문체 5가지](../style/FINETUNE_STYLE_AND_FORMAT.md) · [Colab 학습](../colab/COLAB_TRAINING.md)

---

## 1. 어떤 어댑터인가

| 항목 | 내용 |
|------|------|
| 종류 | **LoRA** (`peft_type: LORA`) |
| 태스크 | **CAUSAL_LM** (디코더-only 생성 모델) |
| 붙는 위치 | 어텐션·MLP 투영층: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| 학습 스크립트 | 표준: `finetune/train_lora.py` · 선택: `finetune/train_lora_unsloth.py` |

**새 어휘 토큰을 추가하지 않습니다.** 토크나이저는 베이스와 동일 계열이며, `pad_token`이 없을 때만 학습 코드에서 `pad_token = eos_token`으로 맞춥니다.

---

## 2. 어댑터 폴더에 저장되는 옵션 (`adapter_config.json`)

학습이 끝나면 출력 디렉터리에 **LoRA 가중치**와 함께 **PEFT 설정**이 저장됩니다. 예시 필드와 의미는 다음과 같습니다.

| 필드 | 의미 |
|------|------|
| `base_model_name_or_path` | **짝이 맞아야 하는 베이스 모델** (다르면 로드·합성 오류 또는 품질 붕괴) |
| `r` | 저랭크 차원 (기본 **16**) |
| `lora_alpha` | 스케일에 쓰이는 α (기본 **32**, 보통 α/r 비율이 스케일에 영향) |
| `lora_dropout` | 드롭아웃 — `train_lora.py`는 코드상 **0.05 고정**, `train_lora_unsloth.py`는 CLI로 조절(기본 **0**) |
| `target_modules` | LoRA가 붙은 모듈 이름 목록 |
| `bias` | 보통 **`none`** (바이어스 미학습) |
| `task_type` | **`CAUSAL_LM`** |

**여기에는 안 남는 것:** 에폭, 배치, learning rate, `max_seq_length`, 저장 전략, W&B, 4bit 여부 등 **학습 런타임 옵션**은 `adapter_config`에 기록되지 않습니다. 재현하려면 노트북·명령어를 별도로 남기는 것이 좋습니다.

---

## 3. 학습 스크립트별 CLI 요약

### `train_lora.py`

| 옵션 | 기본값 | 비고 |
|------|--------|------|
| `--data` | (필수) | `finetune_dataset.jsonl` 등 |
| `--out` | `./lora-output` | 어댑터·토크나이저 저장 |
| `--base-model` | `MLP-KTLim/llama-3-Korean-Bllossom-8B` | |
| `--epochs` | 3 | |
| `--batch-size` | 2 | |
| `--grad-accum` | 8 | |
| `--lr` | 2e-4 | |
| `--max-length` | 2048 | **학습 시 시퀀스 자르기 상한** |
| `--lora-r` | 16 | |
| `--lora-alpha` | 32 | |
| `--use-4bit` | 꺼짐 | 켜면 QLoRA 스타일 4bit 로딩 |

### `train_lora_unsloth.py` (차이 위주)

| 옵션 | 기본값 | 비고 |
|------|--------|------|
| `--epochs` | 3 | |
| `--grad-accum` | 8 | |
| `--use-4bit` | **켜짐** | `--no-use-4bit`로 끔 |
| `--lora-dropout` | 0 | |
| `--warmup-steps` | 50 | |
| `--optim` | adamw_8bit | |
| `--lr-scheduler-type` | cosine | |
| `--save-strategy` / `--save-steps` | steps / 100 | |
| `--report-to` | none | `wandb` 가능 |
| `--push-to-hub` / `--hub-model-id` | 끔 / 빈값 | |

Unsloth 경로는 **`get_peft_model`으로 이미 LoRA가 씌워진 뒤** `SFTTrainer`에 **`peft_config`를 넘기지 않습니다** (표준 스크립트와 구조상 차이).

---

## 4. 어댑터가 띄는 “성격” (행동 편향)

LoRA는 **학습 `output` 분포에 맞춰** 같은 `instruction`에서 응답을 끌어당깁니다.

- **메타데이터 태스크**: `title:` / `bc_id:` / `sc_keyword:` / `4줄 순서·제약**을 더 지키려는 경향.  
- **국회·검토 문서 태스크**: `##` 소제목, 공문체 (`~함`/`~임`), 현안→한계→검토→과제→기대효과 축.  
- **일반 챗**: 데이터에 상대적으로 적으면, “일반 어시스턴트” 성격은 베이스에 더 가깝게 남을 수 있습니다.

즉 **“만능 글쓰기”가 아니라**, **jsonl에 담긴 역할·형식에 특화된 얇은 층**입니다.

---

## 5. 적용 전 vs 적용 후

### 적용 전 (베이스만)

- 베이스 모델(LLama 계열 Bllossom 등)의 **일반 사전학습·채팅 분포**에 가깝게 응답.  
- **메타 4줄**, **특정 국회형 목차**는 프롬프트만으로 일부 흉내는 가능하나, **형식 이탈·구어체·줄 수 초과**가 더 잘 납니다.  
- **같은 토크나이저·같은 채팅 템플릿**을 써도, **업무용 출력 분포는 덜 맞을 수 있음**.

### 적용 후 (베이스 + LoRA)

- **학습에 넣은 `instruction`/태스크에 가까운 형식·톤**으로 수렴.  
- 메타 태스크면 **4줄 계약**이 상대적으로 안정적.  
- 의회형 데이터가 많으면 **공문체·소제목 구조**가 강해짐.  
- **부작용**: 다른 도메인 질문에서 말투가 지나치게 “보고서체”로 굳거나, 베이스만 쓸 때보다 덜 자연스러울 수 있음 (데이터 비중·강도에 따름).

### 적용 후 “어떻게 쓰면 되나” (추론)

1. **베이스 모델**을 `adapter_config.json`의 `base_model_name_or_path`와 **호환되는 것**으로 로드.  
2. **PEFT로 어댑터 로드** 후 병합 또는 어댑터 연결 추론.  
3. **학습 때와 동일한 system prompt** (`instruction`)와 user 포맷을 맞춤 — `train_lora.example_to_text`와 같은 계약이면 재현이 가장 좋음.  
4. 생성 길이는 학습 `max-length`와 별개 — **`max_new_tokens` / `num_predict` 등은 추론 서버에서 설정**.

---

## 6. 자주 나는 실수(파기·불일치)

- **베이스와 어댑터 불일치**: 예) 어댑터는 3B용인데 8B에 붙임 → 오류 또는 품질 저하.  
- **토크나이저/채팅 템플릿 불일치**: 학습은 chat_template으로 직렬화했는데 추론은 다른 프롬프트 형식.  
- **태스크 불일치**: 메타용으로만 학습했는데 장문 요약만 기대함 — 기대치 조정 또는 데이터 추가.

---

## 7. 저장소 내 참고 경로

- 결과 예시: `result/*/adapter_config.json` (실제 베이스 id는 파일을 열어 확인)  
- 학습 진입: `finetune/colab_train.ipynb`, `finetune/kaggle_train.ipynb`, Unsloth 변형 노트북
