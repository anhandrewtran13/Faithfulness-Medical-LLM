#!/usr/bin/env python3
"""QLoRA fine-tuning for RQ1: dense LoRA or MoE-LoRA on a pinned medical MCQ manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from data import TrainingExample, load_training_manifest, subsample_examples
from moe_lora import (
    MoELoraConfig,
    apply_moe_lora,
    load_moe_lora,
    save_moe_lora,
    trainable_moe_lora_params,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MANIFEST = "data/manifests/rq1_training_manifest.jsonl"


@dataclass
class TokenizedBatch:
    input_ids: list[int]
    labels: list[int]


class MedicalMCQDataset(Dataset):
    def __init__(self, examples: list[TrainingExample], tokenizer, max_seq_len: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        messages = [
            {"role": "user", "content": ex.user_text},
            {"role": "assistant", "content": ex.assistant_text},
        ]
        full_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ex.user_text}],
            tokenize=True,
            add_generation_prompt=True,
        )
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        input_ids = full_ids[-self.max_seq_len :]
        labels = labels[-self.max_seq_len :]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }


class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in batch)
        input_ids, labels, attention_mask = [], [], []
        for item in batch:
            pad_len = max_len - item["input_ids"].shape[0]
            input_ids.append(
                torch.cat(
                    [
                        item["input_ids"],
                        torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                    ]
                )
            )
            labels.append(
                torch.cat(
                    [
                        item["labels"],
                        torch.full((pad_len,), -100, dtype=torch.long),
                    ]
                )
            )
            attention_mask.append(
                torch.cat(
                    [
                        item["attention_mask"],
                        torch.zeros(pad_len, dtype=torch.long),
                    ]
                )
            )
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }


class QLoRATrainer(Trainer):
    """Accept custom MoE-LoRA on 4-bit models; skip broken PEFT checkpoint saves."""

    def __init__(self, *args, custom_adapter: bool = False, **kwargs):
        self._custom_adapter = custom_adapter
        model = kwargs.get("model")
        if model is None and args:
            model = args[0]
        restore_peft_flag = False
        if (
            custom_adapter
            and getattr(model, "is_quantized", False)
            and not getattr(model, "_hf_peft_config_loaded", False)
        ):
            model._hf_peft_config_loaded = True
            restore_peft_flag = True
        super().__init__(*args, **kwargs)
        if restore_peft_flag:
            model._hf_peft_config_loaded = False

    def save_model(self, output_dir=None, _internal_call=False):
        if self._custom_adapter:
            return
        super().save_model(output_dir, _internal_call=_internal_call)


def load_base_model(model_name: str, load_in_4bit: bool) -> AutoModelForCausalLM:
    kwargs: dict = dict(
        trust_remote_code=True,
        device_map="auto",
    )
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def prepare_model(
    model_name: str,
    adapter: str,
    lora_rank: int,
    lora_alpha: int,
    num_experts: int,
    load_in_4bit: bool,
    init_adapter_dir: str | None = None,
) -> tuple[AutoModelForCausalLM, list[str] | None, MoELoraConfig | None]:
    model = load_base_model(model_name, load_in_4bit=load_in_4bit)
    model.config.use_cache = False
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    patched_layers: list[str] | None = None
    moe_config: MoELoraConfig | None = None

    if init_adapter_dir:
        init_path = Path(init_adapter_dir)
        if adapter == "dense":
            model = PeftModel.from_pretrained(model, str(init_path), is_trainable=True)
            model.print_trainable_parameters()
            return model, patched_layers, moe_config
        if adapter == "moe":
            model = load_moe_lora(model, str(init_path))
            patched_layers = []
            with open(init_path / "moe_lora_config.json") as f:
                meta = json.load(f)
            moe_config = MoELoraConfig(
                num_experts=meta["num_experts"],
                rank=meta["rank"],
                alpha=meta["alpha"],
                dropout=meta.get("dropout", 0.05),
                target_modules=tuple(meta["target_modules"]),
            )
            patched_layers = meta["patched_layers"]
            for param in model.parameters():
                param.requires_grad = False
            for param in trainable_moe_lora_params(model):
                param.requires_grad = True
            trainable = trainable_moe_lora_params(model)
            total = sum(p.numel() for p in trainable)
            print(
                f"Resumed MoE-LoRA from {init_path}: "
                f"{total:,} trainable params across {len(patched_layers)} layers"
            )
            return model, patched_layers, moe_config
        raise ValueError(f"Cannot resume adapter type {adapter}")

    if adapter == "dense":
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
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
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif adapter == "moe":
        moe_config = MoELoraConfig(num_experts=num_experts, rank=lora_rank, alpha=lora_alpha)
        model, patched_layers = apply_moe_lora(model, moe_config)
        for param in model.parameters():
            param.requires_grad = False
        for param in trainable_moe_lora_params(model):
            param.requires_grad = True
        trainable = trainable_moe_lora_params(model)
        total = sum(p.numel() for p in trainable)
        print(f"MoE-LoRA trainable params: {total:,} across {len(patched_layers)} layers")
    else:
        raise ValueError(f"Unknown adapter type: {adapter}")
    return model, patched_layers, moe_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", choices=["dense", "moe"], required=True)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_samples", type=int, default=0, help="0 = use full manifest.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--no_4bit", action="store_true", help="Disable 4-bit QLoRA.")
    p.add_argument("--run_name", default="")
    p.add_argument(
        "--init_adapter_dir",
        default="",
        help="Optional existing dense/MoE adapter to continue training (abstention recovery).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_in_4bit = args.load_in_4bit and not args.no_4bit
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Training manifest not found: {manifest_path}. "
            "Run scripts/rq1/build_training_manifest.py first."
        )

    examples = load_training_manifest(manifest_path)
    examples = subsample_examples(examples, args.max_samples, seed=args.seed)
    print(f"Training on {len(examples)} examples from {manifest_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, patched_layers, moe_config = prepare_model(
        model_name=args.model,
        adapter=args.adapter,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        num_experts=args.num_experts,
        load_in_4bit=load_in_4bit,
        init_adapter_dir=args.init_adapter_dir or None,
    )

    dataset = MedicalMCQDataset(examples, tokenizer, max_seq_len=args.max_seq_len)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "model": args.model,
        "adapter": args.adapter,
        "manifest": str(manifest_path.resolve()),
        "n_train_examples": len(examples),
        "seed": args.seed,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "num_experts": args.num_experts,
        "max_steps": args.max_steps,
        "load_in_4bit": load_in_4bit,
        "init_adapter_dir": args.init_adapter_dir or None,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
        f.write("\n")

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="no" if args.adapter == "moe" else "steps",
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
        run_name=args.run_name or f"rq1-{args.adapter}",
    )

    trainer = QLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=PadCollator(tokenizer.pad_token_id),
        custom_adapter=(args.adapter == "moe"),
    )
    trainer.train()

    if args.adapter == "dense":
        model.save_pretrained(out_dir / "adapter")
        tokenizer.save_pretrained(out_dir / "adapter")
    else:
        assert patched_layers is not None and moe_config is not None
        save_moe_lora(model, str(out_dir / "adapter"), patched_layers, moe_config)
        tokenizer.save_pretrained(out_dir / "adapter")

    print(f"Saved adapter -> {out_dir / 'adapter'}")


if __name__ == "__main__":
    main()
