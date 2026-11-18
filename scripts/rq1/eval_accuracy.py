#!/usr/bin/env python3
"""Quick MCQ accuracy eval for RQ1 checkpoints on MedQA / MedMCQA / frozen MedXpertQA."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from data import TrainingExample, medmcqa_row_to_example, medqa_row_to_example
from moe_lora import load_moe_lora


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str) -> Path:
    """Resolve data/manifests/... relative to repo root, not cwd."""
    p = Path(path)
    if p.exists():
        return p.resolve()
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    return p


ANSWER_RE = re.compile(r"\(([A-J])\)", re.IGNORECASE)


def parse_answer(text: str) -> str | None:
    matches = ANSWER_RE.findall(text)
    return matches[-1].upper() if matches else None


def load_eval_model(
    base_model: str,
    adapter_dir: str | None,
    adapter_type: str,
    load_in_4bit: bool,
):
    kwargs: dict = dict(trust_remote_code=True, device_map="auto")
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if adapter_dir:
        if adapter_type == "dense":
            model = PeftModel.from_pretrained(model, adapter_dir)
        elif adapter_type == "moe":
            model = load_moe_lora(model, adapter_dir)
        else:
            raise ValueError(adapter_type)
    model.eval()
    return model


@torch.inference_mode()
def predict(model, tokenizer, example: TrainingExample, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": example.user_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True)


def eval_examples(
    model,
    tokenizer,
    examples: list[TrainingExample],
    max_new_tokens: int,
) -> dict:
    correct = 0
    parsed = 0
    for ex in examples:
        text = predict(model, tokenizer, ex, max_new_tokens=max_new_tokens)
        pred = parse_answer(text)
        if pred is not None:
            parsed += 1
        if pred == ex.answer_letter:
            correct += 1
    n = len(examples)
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "parse_rate": parsed / n if n else 0.0,
        "correct": correct,
    }


def load_medxpertqa_manifest(path: str, n_items: int) -> list[TrainingExample]:
    with open(path) as f:
        manifest = json.load(f)
    items = manifest["items"][:n_items] if n_items > 0 else manifest["items"]
    return [
        TrainingExample(
            id=row["qid"],
            source="medxpertqa",
            question=row["question"],
            options=row["options"],
            answer_letter=row["gold"],
        )
        for row in items
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter_dir", default="", help="Empty = evaluate base model only.")
    p.add_argument("--adapter_type", choices=["dense", "moe", "none"], default="none")
    p.add_argument("--benchmarks", nargs="+", default=["medqa", "medmcqa"])
    p.add_argument("--medxpertqa_manifest", default="data/manifests/medxpertqa_reasoning_n200_seed42.json")
    p.add_argument("--n_per_benchmark", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--no_4bit", action="store_true")
    p.add_argument("--out_json", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_in_4bit = args.load_in_4bit and not args.no_4bit
    adapter_type = args.adapter_type
    if not args.adapter_dir and adapter_type != "none":
        adapter_type = "none"

    medxpertqa_manifest: Path | None = None
    if "medxpertqa" in args.benchmarks:
        medxpertqa_manifest = resolve_repo_path(args.medxpertqa_manifest)
        if not medxpertqa_manifest.exists():
            raise FileNotFoundError(
                f"Missing {medxpertqa_manifest}. Run freeze_eval_manifest.py first."
            )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = load_eval_model(
        args.base_model,
        args.adapter_dir or None,
        adapter_type,
        load_in_4bit=load_in_4bit,
    )

    results: dict[str, dict] = {}
    if "medqa" in args.benchmarks:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        examples = []
        for idx, row in enumerate(ds):
            if idx >= args.n_per_benchmark:
                break
            examples.append(medqa_row_to_example(row, idx))
        results["medqa_usmle_test"] = eval_examples(
            model, tokenizer, examples, args.max_new_tokens
        )

    if "medmcqa" in args.benchmarks:
        ds = load_dataset("openlifescienceai/medmcqa", split="validation")
        examples = []
        for row in ds:
            ex = medmcqa_row_to_example(row)
            if ex is not None:
                examples.append(ex)
            if len(examples) >= args.n_per_benchmark:
                break
        results["medmcqa_validation"] = eval_examples(
            model, tokenizer, examples, args.max_new_tokens
        )

    if "medxpertqa" in args.benchmarks:
        assert medxpertqa_manifest is not None
        examples = load_medxpertqa_manifest(
            str(medxpertqa_manifest), n_items=args.n_per_benchmark
        )
        results["medxpertqa_reasoning_n200"] = eval_examples(
            model, tokenizer, examples, args.max_new_tokens
        )

    print(json.dumps(results, indent=2))
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
            f.write("\n")
        print(f"Wrote -> {out}")


if __name__ == "__main__":
    main()
