#!/usr/bin/env python3
"""MetaMedQA metacognition eval via OpenAI or Anthropic API (no GPU deps)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

FRONTIER_DIR = Path(__file__).resolve().parent
RQ2_DIR = FRONTIER_DIR.parent / "rq2"
REPO_ROOT = FRONTIER_DIR.parents[1]
sys.path.insert(0, str(RQ2_DIR))
sys.path.insert(0, str(FRONTIER_DIR))

from api_model import ApiModel  # noqa: E402
from data import (  # noqa: E402
    ABSTAIN_LETTERS,
    MetaMedQAItem,
    items_from_manifest,
    load_manifest,
    resolve_repo_path,
    select_subset,
)
from metrics import confidence_to_unit, expected_calibration_error, sample_consistency  # noqa: E402
from parsing import parse_answer, parse_confidence, parse_consistency_sample, parse_trace  # noqa: E402
from prompts import consistency_prompt, metamedqa_prompt  # noqa: E402


@dataclass
class ItemResult:
    qid: str
    kind: str
    gold: str
    answer: str
    confidence: int
    correct: bool
    abstained: bool
    parse_ok: bool
    trace: str = ""
    consistency_samples: list[str] = field(default_factory=list)
    consistency_score: float = 0.0
    binary_trials: list[dict] = field(default_factory=list)


def evaluate_item_api(
    model: ApiModel,
    item: MetaMedQAItem,
    n_consistency_samples: int,
    max_new_tokens: int,
    consistency_temperature: float,
    prompt_style: str,
) -> ItemResult:
    prompt = metamedqa_prompt(item, style=prompt_style)
    think_prefill = "## Thinking"
    model.chat_prompt(prompt, assistant_prefill=think_prefill)
    out = model.generate(prompt, max_new_tokens=max_new_tokens)
    full_text = think_prefill + "\n" + out
    answer = parse_answer(full_text, valid_letters=item.option_letters)
    confidence = parse_confidence(full_text)

    consistency_samples: list[str] = []
    if n_consistency_samples > 0:
        user_prompt = consistency_prompt(
            item, style=prompt_style, prefill="## Final Response\nAnswer: "
        )
        consistency_tokens = 4096 if model.is_reasoning_model else 64
        for _ in range(n_consistency_samples):
            if model.is_reasoning_model:
                # o-series models ignore assistant prefill; ask for a single letter explicitly.
                letter_prompt = (
                    user_prompt
                    + "\n\nWithout additional reasoning, reply with ONLY the single answer "
                    "letter (A-F), nothing else."
                )
                model.chat_prompt(letter_prompt)
                sample_out = model.generate(
                    letter_prompt,
                    max_new_tokens=consistency_tokens,
                    do_sample=True,
                    temperature=consistency_temperature,
                )
            else:
                model.chat_prompt(user_prompt, assistant_prefill="## Final Response\nAnswer: ")
                sample_out = model.generate(
                    user_prompt,
                    max_new_tokens=consistency_tokens,
                    do_sample=True,
                    temperature=consistency_temperature,
                )
            letter = parse_consistency_sample(
                "## Final Response\nAnswer: " + sample_out, item.option_letters
            )
            consistency_samples.append(letter or "")

    return ItemResult(
        qid=item.qid,
        kind=item.kind,
        gold=item.gold,
        answer=answer or "",
        confidence=confidence,
        correct=answer == item.gold,
        abstained=answer in {item.idk_letter, item.none_of_above_letter},
        parse_ok=bool(answer),
        trace=parse_trace(full_text),
        consistency_samples=consistency_samples,
        consistency_score=sample_consistency(consistency_samples) if consistency_samples else 0.0,
        binary_trials=[],
    )


def aggregate_results(results: list[ItemResult]) -> dict:
    n = len(results)
    correct = sum(1 for r in results if r.correct)
    parsed = sum(1 for r in results if r.parse_ok)
    confidences = [confidence_to_unit(r.confidence) for r in results if r.parse_ok]
    correctness = [r.correct for r in results if r.parse_ok]
    ece = expected_calibration_error(confidences, correctness, n_bins=5)

    abstention_items = [r for r in results if r.gold in ABSTAIN_LETTERS]
    abstention_hits = sum(1 for r in abstention_items if r.abstained)
    unknown_items = [r for r in results if r.kind in {"Bad", "Fictional", "Question modified"}]
    unknown_idk_hits = sum(1 for r in unknown_items if r.answer == "F")
    idk_gold_items = [r for r in results if r.gold == "F"]
    idk_recall = sum(1 for r in idk_gold_items if r.answer == "F")
    missing_gold_items = [r for r in results if r.gold == "E"]
    missing_recall = sum(1 for r in missing_gold_items if r.answer == "E")

    by_kind: dict[str, dict] = defaultdict(dict)
    for kind in sorted({r.kind for r in results}):
        sub = [r for r in results if r.kind == kind]
        by_kind[kind] = {
            "n": len(sub),
            "accuracy": sum(r.correct for r in sub) / len(sub) if sub else 0.0,
            "parse_rate": sum(r.parse_ok for r in sub) / len(sub) if sub else 0.0,
            "mean_confidence": sum(r.confidence for r in sub) / len(sub) if sub else 0.0,
        }

    return {
        "n_items": n,
        "accuracy": correct / n if n else 0.0,
        "parse_rate": parsed / n if n else 0.0,
        "ece": ece,
        "mean_confidence": sum(r.confidence for r in results) / n if n else 0.0,
        "mean_consistency": sum(r.consistency_score for r in results) / n if n else 0.0,
        "abstention_recall": abstention_hits / len(abstention_items) if abstention_items else 0.0,
        "unknown_item_abstention": unknown_idk_hits / len(unknown_items) if unknown_items else 0.0,
        "idk_recall": idk_recall / len(idk_gold_items) if idk_gold_items else 0.0,
        "missing_answer_recall": missing_recall / len(missing_gold_items) if missing_gold_items else 0.0,
        "idk_recall_n": len(idk_gold_items),
        "missing_recall_n": len(missing_gold_items),
        "by_kind": dict(by_kind),
    }


def slugify(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").replace(" ", "_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", choices=["anthropic", "openai"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "data/manifests/metamedqa_eval_seed42.json"),
    )
    p.add_argument("--subset", choices=["all", "eval", "pilot"], default="eval")
    p.add_argument("--pilot_n", type=int, default=0)
    p.add_argument("--out_jsonl", default="")
    p.add_argument("--summary_json", default="")
    p.add_argument("--n_consistency_samples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--consistency_temperature", type=float, default=0.7)
    p.add_argument("--prompt_style", default="metamedqa_full")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--rerun_consistency_from",
        default="",
        help="Reuse cached answers from prior JSONL and only re-run consistency samples.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(resolve_repo_path(args.manifest))
    items = select_subset(items_from_manifest(manifest), args.subset, manifest)
    if args.pilot_n > 0:
        items = items[: args.pilot_n]

    slug = slugify(f"{args.provider}_{args.model}")
    out_jsonl = Path(args.out_jsonl or REPO_ROOT / f"results/metamedqa_frontier_{slug}.jsonl")
    summary_json = Path(
        args.summary_json or str(out_jsonl.with_name(out_jsonl.stem + "_summary.json"))
    )
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    cached_rows: dict[str, dict] = {}
    if args.rerun_consistency_from:
        with open(args.rerun_consistency_from) as f:
            for line in f:
                row = json.loads(line)
                cached_rows[row["qid"]] = row
        print(f"Loaded {len(cached_rows)} rows from {args.rerun_consistency_from}")
        print("Re-running consistency samples only.")

    done: set[str] = set()
    if args.resume and out_jsonl.exists() and not args.rerun_consistency_from:
        with open(out_jsonl) as f:
            for line in f:
                done.add(json.loads(line)["qid"])

    model = ApiModel(args.provider, args.model)
    write_mode = "w" if args.rerun_consistency_from else ("a" if args.resume and out_jsonl.exists() else "w")
    with open(out_jsonl, write_mode) as out_f:
        for i, item in enumerate(items, 1):
            if item.qid in done and not args.rerun_consistency_from:
                continue
            cached = cached_rows.get(item.qid)
            if cached and args.rerun_consistency_from:
                print(f"[{i}/{len(items)}] {item.qid} kind={item.kind} (consistency rerun)")
                row = ItemResult(**{k: cached[k] for k in ItemResult.__dataclass_fields__ if k in cached})
                if args.n_consistency_samples > 0:
                    user_prompt = consistency_prompt(
                        item, style=args.prompt_style, prefill="## Final Response\nAnswer: "
                    )
                    consistency_tokens = 4096 if model.is_reasoning_model else 64
                    samples: list[str] = []
                    for _ in range(args.n_consistency_samples):
                        if model.is_reasoning_model:
                            letter_prompt = (
                                user_prompt
                                + "\n\nWithout additional reasoning, reply with ONLY the single "
                                "answer letter (A-F), nothing else."
                            )
                            model.chat_prompt(letter_prompt)
                            sample_out = model.generate(
                                letter_prompt,
                                max_new_tokens=consistency_tokens,
                                do_sample=True,
                                temperature=args.consistency_temperature,
                            )
                        else:
                            model.chat_prompt(
                                user_prompt, assistant_prefill="## Final Response\nAnswer: "
                            )
                            sample_out = model.generate(
                                user_prompt,
                                max_new_tokens=consistency_tokens,
                                do_sample=True,
                                temperature=args.consistency_temperature,
                            )
                        letter = parse_consistency_sample(
                            "## Final Response\nAnswer: " + sample_out, item.option_letters
                        )
                        samples.append(letter or "")
                    row.consistency_samples = samples
                    row.consistency_score = (
                        sample_consistency(samples) if samples else 0.0
                    )
            else:
                print(f"[{i}/{len(items)}] {item.qid} kind={item.kind}")
                row = evaluate_item_api(
                    model=model,
                    item=item,
                    n_consistency_samples=args.n_consistency_samples,
                    max_new_tokens=args.max_new_tokens,
                    consistency_temperature=args.consistency_temperature,
                    prompt_style=args.prompt_style,
                )
            out_f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            out_f.flush()

    all_rows: list[ItemResult] = []
    with open(out_jsonl) as f:
        for line in f:
            d = json.loads(line)
            all_rows.append(ItemResult(**{k: d[k] for k in ItemResult.__dataclass_fields__}))

    summary = aggregate_results(all_rows)
    summary.update(
        {
            "model": model.name,
            "provider": args.provider,
            "api_model": args.model,
            "manifest": str(resolve_repo_path(args.manifest)),
            "subset": args.subset,
            "prompt_style": args.prompt_style,
        }
    )
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_rows)} items -> {out_jsonl}")
    print(
        f"Acc={summary['accuracy']:.3f}  Unk.Abstain={summary['unknown_item_abstention']:.3f}  "
        f"ECE={summary['ece']:.3f}  Consistency={summary['mean_consistency']:.3f}"
    )


if __name__ == "__main__":
    main()
