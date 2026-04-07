from __future__ import annotations

from unsloth import FastLanguageModel
from transformers import TrainingArguments

import argparse
import inspect
import os
import shutil
import sys
from pathlib import Path

_FINETUNE_DIR = Path(__file__).resolve().parent
if str(_FINETUNE_DIR) not in sys.path:
    sys.path.insert(0, str(_FINETUNE_DIR))

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from train_lora import example_to_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unsloth LoRA SFT (same JSONL as train_lora.py)"
    )
    p.add_argument(
        "--data", type=str, required=True, help="finetune_dataset.jsonl 경로"
    )
    p.add_argument(
        "--out", type=str, default="./lora-output-unsloth", help="어댑터 저장 디렉터리"
    )
    p.add_argument(
        "--base-model",
        type=str,
        default="MLP-KTLim/llama-3-Korean-Bllossom-8B",
        help="베이스 Causal LM",
    )
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="0 권장(Unsloth·속도), 0.05는 train_lora 기본과 유사",
    )
    p.add_argument(
        "--use-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="QLoRA 4bit (기본 True). --no-use-4bit 로 전체 float 학습",
    )
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--optim", type=str, default="adamw_8bit")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lr-scheduler-type", type=str, default="cosine")
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Unsloth는 True일 때 내부적으로 unsloth 체크포인팅 사용",
    )
    p.add_argument(
        "--save-strategy",
        type=str,
        default="steps",
        choices=("no", "steps", "epoch"),
    )
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument(
        "--report-to",
        type=str,
        default="none",
        help="none | wandb | tensorboard 등 (wandb 사용 시 pip install wandb 및 로그인)",
    )
    p.add_argument("--wandb-run-name", type=str, default="bllossom-sft-unsloth")
    p.add_argument(
        "--push-to-hub", action="store_true", help="학습 종료 후 허브에 푸시"
    )
    p.add_argument(
        "--hub-model-id", type=str, default="", help="예: username/repo-name"
    )
    p.add_argument("--fp16", action="store_true", help="bf16 자동 대신 fp16 강제")
    p.add_argument("--bf16", action="store_true", help="fp16 자동 대신 bf16 강제")
    return p.parse_args()


def main() -> None:
    try:
        from unsloth import FastLanguageModel  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit(
            "unsloth 패키지가 없습니다. Colab/Kaggle에서는 finetune/colab_unsloth_train.ipynb "
            "또는 kaggle_unsloth_train.ipynb 의 설치 셀을 참고하세요.\n"
            f"원인: {e}"
        ) from e

    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if not os.path.isfile(args.data):
        raise SystemExit(f"데이터 파일 없음: {args.data}")

    raw = load_dataset("json", data_files=args.data, split="train")
    if len(raw) == 0:
        raise SystemExit("데이터셋이 비어 있습니다.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=args.use_4bit,
        token=hf_token,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    gc_mode: str | bool = "unsloth" if args.gradient_checkpointing else False
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=gc_mode,
        random_state=3407,
    )

    def to_text(batch):
        if isinstance(batch["instruction"], list):
            texts = []
            for i in range(len(batch["instruction"])):
                ex = {
                    "instruction": batch["instruction"][i],
                    "input": batch["input"][i],
                    "output": batch["output"][i],
                }
                texts.append(example_to_text(tokenizer, ex))
            return {"text": texts}
        return {
            "text": example_to_text(
                tokenizer,
                {
                    "instruction": batch["instruction"],
                    "input": batch["input"],
                    "output": batch["output"],
                },
            )
        }

    ds = raw.map(
        to_text,
        batched=True,
        remove_columns=[c for c in raw.column_names if c != "text"],
    )
    if "text" not in ds.column_names:
        raise SystemExit("내부 오류: text 컬럼 생성 실패")

    if args.fp16:
        use_bf16, use_fp16 = False, True
    elif args.bf16:
        use_bf16, use_fp16 = True, False
    else:
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and not use_bf16

    save_steps = args.save_steps if args.save_strategy == "steps" else None

    sft_common: dict = dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy=args.save_strategy,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=args.report_to,
        push_to_hub=args.push_to_hub,
        packing=False,
    )
    if args.report_to == "wandb":
        sft_common["run_name"] = args.wandb_run_name
    if args.save_strategy == "steps" and save_steps is not None:
        sft_common["save_steps"] = save_steps
    if args.push_to_hub:
        hub_id = (args.hub_model_id or "").strip()
        if not hub_id:
            raise SystemExit("--push-to-hub 사용 시 --hub-model-id 를 지정하세요.")
        sft_common["hub_model_id"] = hub_id
        if hf_token:
            sft_common["hub_token"] = hf_token

    cfg_params = set(inspect.signature(SFTConfig.__init__).parameters)
    tr_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    text_seq = {
        "dataset_text_field": "text",
        "max_seq_length": args.max_length,
    }
    on_cfg = {k: v for k, v in text_seq.items() if k in cfg_params}
    on_tr = {k: v for k, v in text_seq.items() if k in tr_params and k not in on_cfg}
    sft_config = SFTConfig(**sft_common, **on_cfg)

    # 이미 get_peft_model 적용됨 — peft_config 는 넘기지 않음 (기존 train_lora.py 와 차이)
    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=ds,
        **on_tr,
    )
    if "processing_class" in tr_params:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    elif "tokenizer" in tr_params:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)
    else:
        trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    for folder in os.listdir(args.out):
        folder_path = os.path.join(args.out, folder)
        if os.path.isdir(folder_path) and folder.startswith("checkpoint-"):
            shutil.rmtree(folder_path)

    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
