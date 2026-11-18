#!/usr/bin/env python3
"""Run MetaMedQA metacognition eval: answer + confidence, DMC, sample-consistency, ECE."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (  # noqa: E402
    ABSTAIN_LETTERS,
    MetaMedQAItem,
    items_from_manifest,
    load_manifest,
    resolve_repo_path,
    select_subset,
)
from metrics import (  # noqa: E402
    BinaryTrial,
    compute_dmc,
    confidence_to_unit,
    expected_calibration_error,
    sample_consistency,
)
from model_utils import (  # noqa: E402
    Model,
    consistency_answer_prefill,
    default_max_new_tokens,
    thinking_prefill,
)
from parsing import (  # noqa: E402
    parse_answer,
    parse_confidence,
    parse_consistency_sample,
    parse_trace,
    parse_trace_or_fallback,
    parse_truncated_answer,
    truncate_words,
)
from prompts import (  # noqa: E402
    PROMPT_STYLES,
    binary_dmc_prompt,
    consistency_prompt,
    metamedqa_prompt,
)


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


def build_binary_trials(item: MetaMedQAItem) -> list[tuple[str, str, str, str]]:
    """Return list of (gold_letter, distractor_letter, option_a, option_b) in A/B form."""
    trials: list[tuple[str, str, str, str]] = []
    for distractor in item.clinical_distractors:
        trials.append((item.gold, distractor, item.gold, distractor))
        trials.append((item.gold, distractor, distractor, item.gold))
    return trials


def run_binary_trial(
    model: Model,
    item: MetaMedQAItem,
    gold_letter: str,
    distractor_letter: str,
    option_a_letter: str,
    option_b_letter: str,
    max_new_tokens: int,
) -> BinaryTrial:
    prompt = binary_dmc_prompt(
        stem=item.question,
        option_a_letter=option_a_letter,
        option_a_text=item.options[option_a_letter],
        option_b_letter=option_b_letter,
        option_b_text=item.options[option_b_letter],
    )
    text = model.generate(model.chat_prompt(prompt), max_new_tokens=max_new_tokens)
    pred = parse_answer(text, valid_letters=["A", "B"])
    confidence = parse_confidence(text)
    gold_ab = "A" if gold_letter == option_a_letter else "B"
    return BinaryTrial(
        qid=item.qid,
        gold=gold_ab,
        pred=pred,
        confidence=confidence,
        correct=pred == gold_ab,
    )


def force_answer_from_partial(
    model: Model,
    item: MetaMedQAItem,
    prompt: str,
    think_prefill: str,
    partial_text: str,
    *,
    truncate_pcts: tuple[int, ...] = (50, 75, 25),
) -> tuple[str, str, int]:
    """Letter-constrained decode after partial CoT; tries several truncation levels."""
    partials: list[str] = []
    base = (partial_text or "").strip()
    if base:
        for pct in truncate_pcts:
            truncated = truncate_words(base, pct)
            if truncated and truncated not in partials:
                partials.append(truncated)
    partials.append("")

    for partial in partials:
        if partial:
            retry_prefill = f"{think_prefill}{partial}\n\n## Final Response\nAnswer: "
        else:
            retry_prefill = f"{think_prefill}\n\n## Final Response\nAnswer: "
        retry_out = model.generate(
            model.chat_prompt(prompt, assistant_prefill=retry_prefill),
            max_new_tokens=48,
            restrict_first_token_to=item.option_letters,
        )
        retry_answer = parse_truncated_answer(retry_out, item.option_letters)
        if retry_answer:
            text = f"{retry_prefill}{retry_out}"
            return retry_answer, text, parse_confidence(text)
    return "", "", 0


def draw_consistency_samples(
    model: Model,
    item: MetaMedQAItem,
    n_samples: int,
    temperature: float,
    prompt_style: str,
    max_new_tokens: int = 1,
) -> list[str]:
    """Stochastic letter samples via prefill + first-token letter constraint."""
    if n_samples <= 0:
        return []
    prefill = consistency_answer_prefill(model.name)
    user_prompt = consistency_prompt(item, style=prompt_style, prefill=prefill)
    prompt = model.chat_prompt(user_prompt, assistant_prefill=prefill)
    letters: list[str] = []
    for _ in range(n_samples):
        sample_text = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            restrict_first_token_to=item.option_letters,
        )
        letters.append(parse_consistency_sample(sample_text, item.option_letters))
    return letters


def evaluate_item(
    model: Model,
    item: MetaMedQAItem,
    n_consistency_samples: int,
    enable_dmc_binary: bool,
    max_new_tokens: int,
    consistency_temperature: float,
    prompt_style: str,
    consistency_max_new_tokens: int = 1,
) -> ItemResult:
    prompt = metamedqa_prompt(item, style=prompt_style)
    think_prefill = thinking_prefill(model.name)
    text = model.generate(
        model.chat_prompt(prompt, assistant_prefill=think_prefill),
        max_new_tokens=max_new_tokens,
    )
    answer = parse_answer(text, valid_letters=item.option_letters)
    confidence = parse_confidence(text)

    if not answer:
        partial = parse_trace_or_fallback(text) or text.strip()
        retry_answer, retry_text, retry_conf = force_answer_from_partial(
            model, item, prompt, think_prefill, partial
        )
        if retry_answer:
            answer = retry_answer
            text = retry_text
            confidence = retry_conf or confidence

    correct = answer == item.gold
    abstained = answer in {item.idk_letter, item.none_of_above_letter}

    consistency_samples: list[str] = []
    if n_consistency_samples > 0:
        consistency_samples = draw_consistency_samples(
            model,
            item,
            n_samples=n_consistency_samples,
            temperature=consistency_temperature,
            prompt_style=prompt_style,
            max_new_tokens=consistency_max_new_tokens,
        )

    binary_rows: list[dict] = []
    binary_trial_objs: list[BinaryTrial] = []
    if enable_dmc_binary:
        for gold_letter, distractor, opt_a, opt_b in build_binary_trials(item):
            trial = run_binary_trial(
                model, item, gold_letter, distractor, opt_a, opt_b, max_new_tokens=512
            )
            binary_trial_objs.append(trial)
            binary_rows.append(asdict(trial))

    return ItemResult(
        qid=item.qid,
        kind=item.kind,
        gold=item.gold,
        answer=answer,
        confidence=confidence,
        correct=correct,
        abstained=abstained,
        parse_ok=bool(answer),
        trace=parse_trace(text),
        consistency_samples=consistency_samples,
        consistency_score=sample_consistency(consistency_samples) if consistency_samples else 0.0,
        binary_trials=binary_rows,
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

    binary_trials = [
        BinaryTrial(
            qid=row["qid"],
            gold=row["gold"],
            pred=row["pred"],
            confidence=row["confidence"],
            correct=row["correct"],
        )
        for r in results
        for row in r.binary_trials
    ]
    dmc = compute_dmc(binary_trials)

    by_kind: dict[str, dict] = defaultdict(dict)
    for kind in sorted({r.kind for r in results}):
        subset = [r for r in results if r.kind == kind]
        by_kind[kind] = {
            "n": len(subset),
            "accuracy": sum(1 for r in subset if r.correct) / len(subset),
            "parse_rate": sum(1 for r in subset if r.parse_ok) / len(subset),
            "mean_confidence": sum(r.confidence for r in subset) / len(subset),
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
        "dmc": {
            "score": None if dmc.note else dmc.dmc_score,
            "d_hat": dmc.d_hat,
            "d_star": dmc.d_star,
            "n_binary_trials": dmc.n_trials,
            "fit_loss": dmc.fit_loss,
            "note": dmc.note,
        },
        "by_kind": dict(by_kind),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="FreedomIntelligence/HuatuoGPT-o1-8B")
    p.add_argument("--base_model", default="", help="Base HF id when --adapter_dir is set.")
    p.add_argument("--adapter_dir", default="", help="LoRA adapter directory (dense or MoE).")
    p.add_argument(
        "--adapter_type",
        choices=["none", "dense", "moe"],
        default="none",
    )
    p.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "data/manifests/metamedqa_eval_seed42.json"),
    )
    p.add_argument("--subset", choices=["all", "eval", "pilot"], default="eval")
    p.add_argument("--out_jsonl", default="")
    p.add_argument("--summary_json", default="")
    p.add_argument("--n_consistency_samples", type=int, default=3)
    p.add_argument("--consistency_max_new_tokens", type=int, default=1)
    p.add_argument("--enable_dmc_binary", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--consistency_temperature", type=float, default=0.7)
    p.add_argument(
        "--prompt_style",
        choices=list(PROMPT_STYLES),
        default="metamedqa_full",
        help="MetaMedQA abstention preamble (default: metamedqa_full per Griot et al.).",
    )
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--no_4bit", action="store_true")
    p.add_argument("--score_only", action="store_true", help="Aggregate metrics from existing JSONL.")
    p.add_argument("--resume", action="store_true", help="Skip qids already present in --out_jsonl.")
    p.add_argument(
        "--rerun_failed",
        action="store_true",
        help="With --resume, re-run items whose cached row has parse_ok=false.",
    )
    p.add_argument(
        "--consistency_only_from",
        default="",
        help="Re-run consistency samples only, using rows from an existing JSONL.",
    )
    args = p.parse_args()
    if not args.out_jsonl and not args.consistency_only_from:
        p.error("--out_jsonl is required unless --consistency_only_from is set.")
    return args


def main() -> None:
    args = parse_args()
    manifest_path = resolve_repo_path(args.manifest)
    manifest = load_manifest(manifest_path)
    all_items = items_from_manifest(manifest)
    items_by_qid = {item.qid: item for item in all_items}
    items = select_subset(all_items, args.subset, manifest)

    if args.consistency_only_from:
        src_path = resolve_repo_path(args.consistency_only_from)
        out_path = Path(args.out_jsonl) if args.out_jsonl else src_path
    else:
        out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.score_only:
        results = [ItemResult(**json.loads(line)) for line in open(out_path)]
    elif args.consistency_only_from:
        load_in_4bit = args.load_in_4bit and not args.no_4bit
        adapter_dir = args.adapter_dir or None
        adapter_type = args.adapter_type if adapter_dir else "none"
        base_model = args.base_model or args.model
        model = Model(
            args.model,
            load_in_4bit=load_in_4bit,
            base_model=base_model,
            adapter_dir=adapter_dir,
            adapter_type=adapter_type,
        )
        results = [ItemResult(**json.loads(line)) for line in open(src_path)]
        print(f"Re-running consistency on {len(results)} items from {src_path}")
        for idx, result in enumerate(results, start=1):
            item = items_by_qid.get(result.qid)
            if item is None:
                print(f"  [{idx}] skip qid={result.qid} (not in manifest subset)")
                continue
            print(f"  [{idx}/{len(results)}] {result.qid} ({result.kind})")
            result.consistency_samples = draw_consistency_samples(
                model,
                item,
                n_samples=args.n_consistency_samples,
                temperature=args.consistency_temperature,
                prompt_style=args.prompt_style,
                max_new_tokens=args.consistency_max_new_tokens,
            )
            result.consistency_score = (
                sample_consistency(result.consistency_samples)
                if result.consistency_samples
                else 0.0
            )
        with open(out_path, "w") as f:
            for result in results:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    else:
        load_in_4bit = args.load_in_4bit and not args.no_4bit
        adapter_dir = args.adapter_dir or None
        adapter_type = args.adapter_type if adapter_dir else "none"
        base_model = args.base_model or args.model
        model = Model(
            args.model,
            load_in_4bit=load_in_4bit,
            base_model=base_model,
            adapter_dir=adapter_dir,
            adapter_type=adapter_type,
        )

        done_qids: set[str] = set()
        results: list[ItemResult] = []
        mode = "a" if args.resume and out_path.exists() else "w"
        if args.resume and out_path.exists():
            with open(out_path) as f:
                for line in f:
                    row = json.loads(line)
                    if args.rerun_failed and not row.get("parse_ok", True):
                        continue
                    done_qids.add(row["qid"])
                    results.append(ItemResult(**row))
            print(f"Resuming: {len(done_qids)} items already in {out_path}")

        effective_max_tokens = default_max_new_tokens(args.model, args.max_new_tokens)
        if effective_max_tokens != args.max_new_tokens:
            print(f"Using max_new_tokens={effective_max_tokens} for {args.model}")

        with open(out_path, "w" if args.rerun_failed and out_path.exists() else mode) as f:
            if args.rerun_failed and out_path.exists():
                for result in results:
                    f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            for idx, item in enumerate(items, start=1):
                if item.qid in done_qids:
                    continue
                print(f"[{idx}/{len(items)}] {item.qid} ({item.kind})")
                result = evaluate_item(
                    model,
                    item,
                    n_consistency_samples=args.n_consistency_samples,
                    enable_dmc_binary=args.enable_dmc_binary,
                    max_new_tokens=effective_max_tokens,
                    consistency_temperature=args.consistency_temperature,
                    prompt_style=args.prompt_style,
                    consistency_max_new_tokens=args.consistency_max_new_tokens,
                )
                results.append(result)
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                f.flush()

    summary = aggregate_results(results)
    summary["model"] = args.model
    summary["base_model"] = args.base_model or args.model
    summary["adapter_dir"] = args.adapter_dir or None
    summary["adapter_type"] = args.adapter_type if args.adapter_dir else "none"
    summary["manifest"] = str(manifest_path)
    summary["subset"] = args.subset
    summary["prompt_style"] = args.prompt_style
    print(json.dumps(summary, indent=2))

    summary_path = Path(args.summary_json) if args.summary_json else out_path.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
