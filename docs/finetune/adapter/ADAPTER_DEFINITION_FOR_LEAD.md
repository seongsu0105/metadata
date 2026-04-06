이 문서는 팀 **리드·조장에게 붙여넣기용**으로, Colab `train_lora.py` 학습 산출물(메타데이터 LoRA)을 짧게 정의한 것이다.

LoRA 어댑터 정의 (리드·조장 공유, 복사용)

colab_train.ipynb 로 train_lora.py 학습을 끝냈을 때 나오는 산출물 정의.
코드에 출력 형식이 하드코딩되지 않음. 데이터 JSONL 의 instruction·output 이 계약이다.

한 줄: 베이스 MLP-KTLim/llama-3-Korean-Bllossom-8B 에 붙는 PEFT LoRA(SFT) 어댑터. 추론 시 베이스와 adapter_config 의 base 경로·토크나이저·챗 템플릿을 짝지울 것.

산출물 위치: OUTPUT_DIR (Colab 기본 /content/lora-output). 내용: LoRA 가중치, adapter_config.json, 토크나이저.

종류: PEFT LORA, task_type CAUSAL_LM.

target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

train_lora.py 코드 고정값 예: lora_dropout 0.05, r 16, lora_alpha 32, lr 2e-4, max_length 2048.

colab_train.ipynb 는 기본으로 train_lora.py와 같은 스케줄(epochs 3, grad-accum 8, batch 2)을 넘김. USE_4BIT True 이면 --use-4bit

adapter_config.json 에는 안 남음: 전체 에폭·lr·max_length·4bit 여부 등. 재현하려면 노트북·로그 보관.

데이터 경로 (노트북 기본): finetune_dataset.jsonl

형식: 본 저장소 주류는 시스템 instruction 기준 네 줄만, 순서 title, bc_id, sc_keyword, tl_summary. 같은 파일에 instruction 이 다른 행이 섞이면 한 어댑터가 모두 함께 학습하고 비율은 행 개수로 결정. [안건분류]→[심사요약] 같은 고정 네 라벨은 메타 instruction 에 없음.

문체: 메타 instruction 은 인사·근거 밖 추정 금지. 국회형 output 이 많으면 검토서체(~함·~임·현행 제도는·…할 필요가 있음 등)로 치우침. 사료됨·가결·부결 등이 output 에 거의 없으면 그 말투는 중심으로 학습되지 않음.

Reduce: 메타는 긴 입력을 제목·키워드·짧은 요약으로 압축. 국회형 행이 있으면 메모·불릿·대화를 단락·불릿·소제목 순으로 재구성 패턴도 함께 학습. 데이터에 없는 관용구는 보장 없음.

적용 전: 베이스만. 형식 이탈이 상대적으로 잦을 수 있음.

적용 후: 베이스+LoRA. 학습에 가까운 instruction 이면 계약에 맞춰 출력이 정렬됨. 학습 데이터에 없는 표현·의결 문구는 약함.

참고 파일: `finetune/train_lora.py`, `finetune/colab_train.ipynb`, [LORA_ADAPTER_GUIDE.md](LORA_ADAPTER_GUIDE.md)
