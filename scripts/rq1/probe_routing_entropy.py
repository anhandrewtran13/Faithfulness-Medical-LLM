#!/usr/bin/env python3
"""Probe MoE-LoRA routing entropy vs MetaMedQA item outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

RQ1_DIR = Path(__file__).resolve().parent
RQ2_DIR = RQ1_DIR.parent / "rq2"
REPO_ROOT = RQ1_DIR.parents[1]
sys.path.insert(0, str(RQ1_DIR))
sys.path.insert(0, str(RQ2_DIR))

from data import items_from_manifest, load_manifest, resolve_repo_path  # noqa: E402
from moe_lora import MoELoraLinear, load_moe_lora  # noqa: E402
from prompts import metamedqa_prompt  # noqa: E402


def routing_entropy(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> float:
    """Mean token-level routing entropy averaged across MoE layers."""
    entropies: list[float] = []

    def hook_fn(_module, _inputs, _output):
        return None

    handles = []
    layer_entropies: dict[str, list[float]] = {}

    def make_hook(name: str):
        def _hook(module, inputs, _output):
            x = inputs[0]
            with torch.no_grad():
                gates = module.routing_weights(x)
                # gates: (batch, seq, experts)
                ent = -(gates * (gates + 1e-9).log()).sum(dim=-1).mean().item()
                layer_entropies.setdefault(name, []).append(ent)

        return _hook

    for name, module in model.named_modules():
        if isinstance(module, MoELoraLinear):
            handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    for h in handles:
        h.remove()

    if not layer_entropies:
        return float("nan")
    per_layer = [sum(v) / len(v) for v in layer_entropies.values()]
    return sum(per_layer) / len(per_layer)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter_dir", default=str(REPO_ROOT / "checkpoints/rq3/moe_lora/adapter"))
    p.add_argument(
        "--results_jsonl",
        default=str(REPO_ROOT / "results/metamedqa_rq3_moe_lora.jsonl"),
    )
    p.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "data/manifests/metamedqa_eval_seed42.json"),
    )
    p.add_argument("--out_json", default=str(REPO_ROOT / "results/rq3_moe_routing_entropy.json"))
    p.add_argument("--max_seq_len", type=int, default=1024)
    args = p.parse_args()

    manifest = load_manifest(resolve_repo_path(args.manifest))
    items = {item.qid: item for item in items_from_manifest(manifest)}

    outcomes: dict[str, dict] = {}
    with open(args.results_jsonl) as f:
        for line in f:
            row = json.loads(line)
            outcomes[row["qid"]] = row

    kwargs: dict = dict(trust_remote_code=True, device_map="auto")
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs)
    model = load_moe_lora(model, args.adapter_dir)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    rows: list[dict] = []
    for qid, item in items.items():
        if qid not in outcomes:
            continue
        prompt = metamedqa_prompt(item, style="metamedqa_full")
        messages = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        input_ids = torch.tensor([ids[-args.max_seq_len :]], device=model.device)
        attention_mask = torch.ones_like(input_ids)
        ent = routing_entropy(model, input_ids, attention_mask)
        out = outcomes[qid]
        rows.append(
            {
                "qid": qid,
                "kind": item.kind,
                "correct": out["correct"],
                "abstained": out.get("abstained", False),
                "routing_entropy": ent,
            }
        )
        print(f"{qid} kind={item.kind} correct={out['correct']} entropy={ent:.4f}")

    correct_ent = [r["routing_entropy"] for r in rows if r["correct"]]
    wrong_ent = [r["routing_entropy"] for r in rows if not r["correct"]]
    summary = {
        "n_items": len(rows),
        "mean_entropy_correct": sum(correct_ent) / len(correct_ent) if correct_ent else None,
        "mean_entropy_wrong": sum(wrong_ent) / len(wrong_ent) if wrong_ent else None,
        "items": rows,
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
