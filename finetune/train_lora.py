#!/usr/bin/env python3
"""
finetune_dataset.jsonl (instruction / input / output) → LoRA 어댑터 학습.

Colab / Kaggle 예:
  python train_lora.py --data /content/finetune_dataset.jsonl --out ./lora-out
  python train_lora.py --data /kaggle/input/your-dataset/finetune_dataset.jsonl --out /kaggle/working/lora-out

Llama 등 게이트 모델은 Hugging Face 토큰이 필요할 수 있음 (Colab 비밀 / Kaggle Secrets):
  HF_TOKEN
"""

from __future__ import annotations

import argparse
import inspect
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA SFT for metadata JSONL")
    p.add_argument(
        "--data",
        type=str,
        required=True,
        help="finetune_dataset.jsonl 경로",
    )
    p.add_argument(
        "--out",
        type=str,
        default="./lora-output",
        help="어댑터·체크포인트 저장 디렉터리",
    )
    p.add_argument(
        "--base-model",
        type=str,
        default="MLP-KTLim/llama-3-Korean-Bllossom-8B",
        help="베이스 Causal LM (허브 id 또는 로컬 경로)",
    )
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument(
        "--use-4bit",
        action="store_true",
        help="QLoRA-style 4bit 로딩 (VRAM 절약, bitsandbytes 필요)",
    )
    return p.parse_args()


def example_to_text(tokenizer, example: dict) -> str:
    inst = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    out = (example.get("output") or "").strip()
    messages = [
        {"role": "system", "content": inst},
        {"role": "user", "content": inp},
        {"role": "assistant", "content": out},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return f"### 시스템\n{inst}\n\n### 사용자\n{inp}\n\n### 어시스턴트\n{out}"


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")

    if not os.path.isfile(args.data):
        raise SystemExit(f"데이터 파일 없음: {args.data}")

    raw = load_dataset("json", data_files=args.data, split="train")
    if len(raw) == 0:
        raise SystemExit("데이터셋이 비어 있습니다.")

    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            ),
            bnb_4bit_use_double_quant=True,
        )

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=dtype if bnb_config is None else None,
        trust_remote_code=True,
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.use_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

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

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    sft_common = dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        packing=False,
    )
    # TRL 버전마다 dataset_text_field / max_seq_length 가 SFTConfig 또는 SFTTrainer 한쪽에만 있다.
    cfg_params = set(inspect.signature(SFTConfig.__init__).parameters)
    tr_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    text_seq = {
        "dataset_text_field": "text",
        "max_seq_length": args.max_length,
    }
    on_cfg = {k: v for k, v in text_seq.items() if k in cfg_params}
    on_tr = {
        k: v
        for k, v in text_seq.items()
        if k in tr_params and k not in on_cfg
    }
    sft_config = SFTConfig(**sft_common, **on_cfg)
    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=ds,
        peft_config=lora,
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
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
