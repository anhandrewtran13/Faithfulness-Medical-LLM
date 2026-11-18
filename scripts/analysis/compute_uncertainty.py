#!/usr/bin/env python3
"""Compute Wilson CIs, answerable-item abstention, and paired McNemar tests.

Reads frozen JSONL result files and writes:
  - results/uncertainty_aggregate.json
  - results/uncertainty_tables.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_stats import McNemarResult, ProportionCI, mcnemar_test, paired_mcnemar, wilson_ci

REPO_ROOT = Path(__file__).resolve().parents[2]

UNKNOWN_KINDS = frozenset({"Bad", "Fictional", "Question modified"})
ANSWERABLE_KINDS = frozenset({"Original", "Answer modified"})
ABSTAIN_ANSWERS = frozenset({"E", "F"})


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def faithfulness_metrics(rows: list[dict], truncation_ks: tuple[int, ...] = (25, 50, 75, 90)) -> dict:
    n = len(rows)
    correct = sum(1 for r in rows if r.get("correct_full"))
    metrics: dict = {
        "n_items": n,
        "accuracy": wilson_ci(correct, n).fmt_ci(),
        "accuracy_raw": wilson_ci(correct, n).__dict__,
    }
    for k in truncation_ks:
        key = str(k)
        hits = sum(
            1
            for r in rows
            if r.get("answer_full")
            and r.get("truncated", {}).get(key) == r.get("answer_full")
        )
        metrics[f"EAR_{k}"] = wilson_ci(hits, n).fmt_ci()
        metrics[f"EAR_{k}_raw"] = wilson_ci(hits, n).__dict__

    correct_rows = [r for r in rows if r.get("correct_full")]
    n_correct = len(correct_rows)
    flipped = sum(1 for r in correct_rows if r.get("biased_flipped"))
    if n_correct:
        metrics["BIFR"] = wilson_ci(flipped, n_correct).fmt_ci()
        metrics["BIFR_raw"] = wilson_ci(flipped, n_correct).__dict__
    else:
        metrics["BIFR"] = "N/A"
        metrics["BIFR_raw"] = None

    ps_rows = [r for r in rows if r.get("paraphrase_stable") is not None]
    if ps_rows:
        stable = sum(1 for r in ps_rows if r.get("paraphrase_stable"))
        metrics["PS"] = wilson_ci(stable, len(ps_rows)).fmt_ci()
        metrics["PS_raw"] = wilson_ci(stable, len(ps_rows)).__dict__
        metrics["PS_n"] = len(ps_rows)
    else:
        metrics["PS"] = "N/A"
        metrics["PS_raw"] = None
        metrics["PS_n"] = 0

    return metrics


def metacognition_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(1 for r in rows if r.get("correct"))
    unknown = [r for r in rows if r.get("kind") in UNKNOWN_KINDS]
    unknown_abstain = sum(1 for r in unknown if r.get("answer") == "F")
    answerable = [r for r in rows if r.get("kind") in ANSWERABLE_KINDS]
    false_abstain = sum(
        1 for r in answerable if r.get("answer") in ABSTAIN_ANSWERS or r.get("abstained")
    )
    original = [r for r in rows if r.get("kind") == "Original"]
    original_false_abstain = sum(
        1 for r in original if r.get("answer") in ABSTAIN_ANSWERS or r.get("abstained")
    )

    out = {
        "n_items": n,
        "accuracy": wilson_ci(correct, n).fmt_ci(),
        "accuracy_raw": wilson_ci(correct, n).__dict__,
        "unknown_item_abstention": (
            wilson_ci(unknown_abstain, len(unknown)).fmt_ci() if unknown else "N/A"
        ),
        "unknown_item_abstention_raw": (
            wilson_ci(unknown_abstain, len(unknown)).__dict__ if unknown else None
        ),
        "unknown_n": len(unknown),
        "answerable_abstention": (
            wilson_ci(false_abstain, len(answerable)).fmt_ci() if answerable else "N/A"
        ),
        "answerable_abstention_raw": (
            wilson_ci(false_abstain, len(answerable)).__dict__ if answerable else None
        ),
        "answerable_n": len(answerable),
        "original_false_abstention": (
            wilson_ci(original_false_abstain, len(original)).fmt_ci() if original else "N/A"
        ),
        "original_false_abstention_raw": (
            wilson_ci(original_false_abstain, len(original)).__dict__ if original else None
        ),
        "original_n": len(original),
    }
    return out


def mcnemar_dict(result: McNemarResult) -> dict:
    return {
        "n_discordant": result.n_discordant,
        "b": result.b,
        "c": result.c,
        "chi2": result.chi2,
        "p_value": result.p_value,
        "significant_005": result.significant_005,
    }


def compare_faithfulness(path_a: Path, path_b: Path, label_a: str, label_b: str) -> dict:
    rows_a = {r["qid"]: r for r in load_jsonl(path_a)}
    rows_b = {r["qid"]: r for r in load_jsonl(path_b)}
    shared = sorted(set(rows_a) & set(rows_b))

    acc_a = {q: bool(rows_a[q].get("correct_full")) for q in shared}
    acc_b = {q: bool(rows_b[q].get("correct_full")) for q in shared}
    acc_test = paired_mcnemar(acc_a, acc_b)

    # BIFR on items correct for both models at baseline
    bifr_shared = [
        q
        for q in shared
        if rows_a[q].get("correct_full") and rows_b[q].get("correct_full")
    ]
    flip_a = {q: bool(rows_a[q].get("biased_flipped")) for q in bifr_shared}
    flip_b = {q: bool(rows_b[q].get("biased_flipped")) for q in bifr_shared}
    bifr_test = paired_mcnemar(flip_a, flip_b) if bifr_shared else mcnemar_test(0, 0)

    return {
        "comparison": f"{label_a} vs {label_b}",
        "n_shared": len(shared),
        "n_bifr_shared_correct": len(bifr_shared),
        "accuracy_mcnemar": mcnemar_dict(acc_test),
        "bifr_mcnemar": mcnemar_dict(bifr_test),
    }


def compare_metacognition(path_a: Path, path_b: Path, label_a: str, label_b: str) -> dict:
    rows_a = {r["qid"]: r for r in load_jsonl(path_a)}
    rows_b = {r["qid"]: r for r in load_jsonl(path_b)}
    shared = sorted(set(rows_a) & set(rows_b))

    acc_a = {q: bool(rows_a[q].get("correct")) for q in shared}
    acc_b = {q: bool(rows_b[q].get("correct")) for q in shared}

    unknown = [q for q in shared if rows_a[q].get("kind") in UNKNOWN_KINDS]
    unk_a = {q: rows_a[q].get("answer") == "F" for q in unknown}
    unk_b = {q: rows_b[q].get("answer") == "F" for q in unknown}

    ans = [q for q in shared if rows_a[q].get("kind") in ANSWERABLE_KINDS]
    fa_a = {
        q: rows_a[q].get("answer") in ABSTAIN_ANSWERS or rows_a[q].get("abstained")
        for q in ans
    }
    fa_b = {
        q: rows_b[q].get("answer") in ABSTAIN_ANSWERS or rows_b[q].get("abstained")
        for q in ans
    }

    return {
        "comparison": f"{label_a} vs {label_b}",
        "n_shared": len(shared),
        "accuracy_mcnemar": mcnemar_dict(paired_mcnemar(acc_a, acc_b)),
        "unknown_abstention_mcnemar": mcnemar_dict(paired_mcnemar(unk_a, unk_b)),
        "answerable_false_abstention_mcnemar": mcnemar_dict(paired_mcnemar(fa_a, fa_b)),
        "n_unknown": len(unknown),
        "n_answerable": len(ans),
    }


def default_configs() -> dict:
    results = REPO_ROOT / "results"
    return {
        "faithfulness": {
            "HuatuoGPT-o1-8B": results / "n200_huatuogpt-o1-8b_fixed.jsonl",
            "m1-7B-23K": results / "n200_m1-7b-23k_fixed.jsonl",
            "DeepSeek-R1-Distill-8B": results / "n200_deepseek-r1-distill-llama-8b_fixed.jsonl",
            "RQ3-Dense": results / "n200_rq3_dense_faithfulness.jsonl",
            "RQ3-MoE": results / "n200_rq3_moe_faithfulness.jsonl",
            "Qwen2.5-7B-Instruct": results / "n200_qwen2.5-7b-instruct_faithfulness.jsonl",
            "RQ2-Dense": results / "n200_rq2_dense_faithfulness.jsonl",
            "RQ2-MoE": results / "n200_rq2_moe_faithfulness.jsonl",
        },
        "paraphrase": {
            "HuatuoGPT-o1-8B": results / "n50_huatuogpt-o1-8b_paraphrase.jsonl",
            "m1-7B-23K": results / "n50_m1-7b-23k_paraphrase.jsonl",
        },
        "metacognition": {
            "HuatuoGPT-o1-8B": results / "metamedqa_FreedomIntelligence_HuatuoGPT-o1-8B.jsonl",
            "Qwen2.5-7B (Phase-2)": results / "metamedqa_Qwen_Qwen2.5-7B-Instruct.jsonl",
            "Qwen2.5-7B (RQ2 base)": results / "metamedqa_rq2_qwen_base.jsonl",
            "m1-7B-23K": results / "metamedqa_UCSC-VLAA_m1-7B-23K.jsonl",
            "RQ2-Dense": results / "metamedqa_rq2_dense_lora.jsonl",
            "RQ2-MoE": results / "metamedqa_rq2_moe_lora.jsonl",
            "RQ3-Dense": results / "metamedqa_rq3_dense_lora.jsonl",
            "RQ3-MoE": results / "metamedqa_rq3_moe_lora.jsonl",
        },
    }


def render_tables(aggregate: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("PHASE 1 FAITHFULNESS (Wilson 95% CI)")
    lines.append("=" * 72)
    for name, m in aggregate["faithfulness"].items():
        lines.append(f"\n{name}:")
        lines.append(f"  Acc:    {m['accuracy']}")
        lines.append(f"  EAR_50: {m.get('EAR_50', 'N/A')}")
        lines.append(f"  EAR_90: {m.get('EAR_90', 'N/A')}")
        lines.append(f"  BIFR:   {m.get('BIFR', 'N/A')}")
        if m.get("PS") != "N/A":
            lines.append(f"  PS (n={m.get('PS_n', 0)}): {m['PS']}")

    lines.append("\n" + "=" * 72)
    lines.append("PHASE 2 METACOGNITION (Wilson 95% CI)")
    lines.append("=" * 72)
    for name, m in aggregate["metacognition"].items():
        lines.append(f"\n{name}:")
        lines.append(f"  Acc:                 {m['accuracy']}")
        lines.append(
            f"  Unk. abstention (n={m['unknown_n']}): {m['unknown_item_abstention']}"
        )
        lines.append(
            f"  Answerable false abst. (n={m['answerable_n']}): {m['answerable_abstention']}"
        )
        lines.append(
            f"  Original false abst. (n={m['original_n']}): {m['original_false_abstention']}"
        )

    lines.append("\n" + "=" * 72)
    lines.append("PAIRED McNEMAR TESTS")
    lines.append("=" * 72)
    for test in aggregate["paired_tests"]:
        lines.append(f"\n{test['comparison']}:")
        for key, val in test.items():
            if key.endswith("_mcnemar") and isinstance(val, dict):
                label = key.replace("_mcnemar", "").replace("_", " ")
                lines.append(
                    f"  {label}: b={val['b']}, c={val['c']}, "
                    f"chi2={val['chi2']:.3f}, p={val['p_value']:.4f}"
                )

    if aggregate.get("qwen_base_medxpertqa"):
        q = aggregate["qwen_base_medxpertqa"]
        lines.append("\n" + "=" * 72)
        lines.append("QWEN BASE MedXpertQA (M1 diagnosis)")
        lines.append("=" * 72)
        if fh := q.get("faithfulness_harness"):
            lines.append(
                f"  Faithfulness harness: {fh['accuracy']} ({fh['correct']}/{fh['n']})"
            )
            lines.append(f"    BIFR: {fh.get('bifr', 'N/A')}")
        if leg := q.get("eval_accuracy_py"):
            lines.append(
                f"  Legacy eval_accuracy.py: {leg['accuracy']} "
                f"({leg['correct']}/{leg['n']}), parse={leg['parse_rate']}"
            )
        lines.append(f"  Note: {q.get('note', '')}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_json",
        default=str(REPO_ROOT / "results/uncertainty_aggregate.json"),
    )
    parser.add_argument(
        "--out_table",
        default=str(REPO_ROOT / "results/uncertainty_tables.txt"),
    )
    args = parser.parse_args()

    configs = default_configs()
    aggregate: dict = {
        "faithfulness": {},
        "metacognition": {},
        "paired_tests": [],
    }

    for name, path in configs["faithfulness"].items():
        if path.exists():
            aggregate["faithfulness"][name] = faithfulness_metrics(load_jsonl(path))

    paraphrase = configs.get("paraphrase", {})
    for name, path in paraphrase.items():
        if path.exists() and name in aggregate["faithfulness"]:
            rows = load_jsonl(path)
            ps_rows = [r for r in rows if r.get("paraphrase_stable") is not None]
            if ps_rows:
                stable = sum(1 for r in ps_rows if r.get("paraphrase_stable"))
                ci = wilson_ci(stable, len(ps_rows))
                aggregate["faithfulness"][name]["PS"] = ci.fmt_ci()
                aggregate["faithfulness"][name]["PS_raw"] = ci.__dict__
                aggregate["faithfulness"][name]["PS_n"] = len(ps_rows)

    for name, path in configs["metacognition"].items():
        if path.exists():
            aggregate["metacognition"][name] = metacognition_metrics(load_jsonl(path))

    faith = configs["faithfulness"]
    meta = configs["metacognition"]

    if faith["HuatuoGPT-o1-8B"].exists() and faith["m1-7B-23K"].exists():
        aggregate["paired_tests"].append(
            compare_faithfulness(
                faith["HuatuoGPT-o1-8B"],
                faith["m1-7B-23K"],
                "HuatuoGPT",
                "m1",
            )
        )

    if meta["Qwen2.5-7B (RQ2 base)"].exists() and meta["RQ2-Dense"].exists():
        aggregate["paired_tests"].append(
            compare_metacognition(
                meta["Qwen2.5-7B (RQ2 base)"],
                meta["RQ2-Dense"],
                "Qwen base",
                "RQ2-Dense",
            )
        )

    if meta["RQ2-Dense"].exists() and meta["RQ3-Dense"].exists():
        aggregate["paired_tests"].append(
            compare_metacognition(
                meta["RQ2-Dense"],
                meta["RQ3-Dense"],
                "RQ2-Dense",
                "RQ3-Dense",
            )
        )

    if meta["HuatuoGPT-o1-8B"].exists() and meta["m1-7B-23K"].exists():
        aggregate["paired_tests"].append(
            compare_metacognition(
                meta["HuatuoGPT-o1-8B"],
                meta["m1-7B-23K"],
                "HuatuoGPT",
                "m1",
            )
        )

    base_eval = REPO_ROOT / "checkpoints/rq1/base/eval.json"
    qwen_faith = REPO_ROOT / "results/n200_qwen2.5-7b-instruct_faithfulness.jsonl"
    if base_eval.exists() or qwen_faith.exists():
        qwen_block: dict = {}
        if qwen_faith.exists():
            qrows = load_jsonl(qwen_faith)
            qcorrect = sum(1 for r in qrows if r.get("correct_full"))
            qn = len(qrows)
            correct_rows = [r for r in qrows if r.get("correct_full")]
            qbifr = sum(1 for r in correct_rows if r.get("biased_flipped"))
            qwen_block["faithfulness_harness"] = {
                "accuracy": wilson_ci(qcorrect, qn).fmt_ci(),
                "correct": qcorrect,
                "n": qn,
                "bifr": wilson_ci(qbifr, len(correct_rows)).fmt_ci() if correct_rows else "N/A",
                "source": str(qwen_faith),
            }
        if base_eval.exists():
            ev = json.loads(base_eval.read_text())
            mx = ev["medxpertqa_reasoning_n200"]
            qwen_block["eval_accuracy_py"] = {
                "accuracy": wilson_ci(mx["correct"], mx["n"]).fmt_ci(),
                "parse_rate": f"{mx['parse_rate']:.3f}",
                "correct": mx["correct"],
                "n": mx["n"],
                "note": "Legacy RQ2 quick-MCQ eval (eval_accuracy.py); 39% parse rate.",
            }
        qwen_block["note"] = (
            "Faithfulness harness reports at-chance board accuracy; "
            "legacy eval_accuracy.py under-reported due to parse failure."
        )
        aggregate["qwen_base_medxpertqa"] = qwen_block

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)
        f.write("\n")

    table_text = render_tables(aggregate)
    out_table = Path(args.out_table)
    with open(out_table, "w") as f:
        f.write(table_text)

    print(table_text)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_table}")


if __name__ == "__main__":
    main()
