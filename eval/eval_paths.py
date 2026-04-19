"""
eval/eval_paths.py
==================
Stage 2a — Path-retrieval evaluation for the GCR pipeline.

Scores the raw beam paths produced by generate_predicted_paths.py directly
against gold answers and gold path sequences.  No LLM is loaded or called;
this script completes in seconds and can be run independently of the
(expensive) answer-generation evaluation in eval_answers.py.

Why a separate script?
----------------------
Path-retrieval quality and answer-generation quality are conceptually
distinct:  one measures whether the constrained decoder finds the correct
reasoning trace in the graph; the other measures whether the LLM converts
that trace into a correct natural-language answer.  Separating them makes
the ablation cleaner and avoids loading a GPU-resident model just to score
string overlap (Luo et al., 2025, §4.2).

Systems
-------
gcr_constrained_paths   — constrained GCR beam paths scored against gold
gcr_unconstrained_paths — unconstrained GCR beam paths scored against gold

Metrics
-------
next_step : EM, token F1, ROUGE-L F1, MRR (over beams),
            path_recall, path_precision
all       : answer_s = 0.0 (no LLM latency)

Output files
------------
<out_dir>/path_answers.jsonl     one record per (instance, system)
<out_dir>/path_results.csv       aggregated metrics
<out_dir>/path_results.tex       LaTeX table

Usage
-----
    python -m eval.eval_paths \\
        --dataset             eval/sampled_100.json \\
        --constrained_paths   results/predicted_paths_constrained.jsonl \\
        --unconstrained_paths results/predicted_paths_unconstrained.jsonl \\
        --out_dir             results

    Quick test (5 instances):
        --limit 5

References
----------
Luo, L., Zhao, Z., Haffari, G., Li, Y.-F., Gong, C., & Pan, S. (2025).
    Graph-constrained reasoning: Faithful reasoning on knowledge graphs
    with large language models. ICML 2025.
Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016).
    SQuAD: 100,000+ questions for machine comprehension of text. EMNLP.
Lin, C.-Y. (2004).
    ROUGE: A package for automatic evaluation of summaries.
    ACL Workshop on Text Summarisation Branches Out.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import json

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eval.metrics import (
    best_path_metrics,
    calculate_path_metrics,
    load_jsonl,
    load_dataset,
    load_done,
    append_record,
    score_answer,
    calculate_path_recall,
    calculate_path_precision,
    aggregate,
    print_results_table,
    write_results_table,
    PATH_METRIC_COLS,
)


# ===========================================================================
# Path-only scoring
# ===========================================================================

def score_paths_directly(
    beams: List[str],
    gold_answer: str,
    question_family: str,
    gold_paths: Optional[List] = None,
    context_block: str = "",
) -> Tuple[str, Dict]:
    """
    Score raw GCR beam paths against *gold_answer* without calling an LLM.

    The prediction string is the concatenation of all non-empty beams, which
    maximises recall and mirrors how a downstream reader would consume the
    full beam set.  MRR is computed over the individual beams, consistent
    with the beam-level ranking used for the LLM-backed GCR systems.

    This two-stage decomposition — measuring path-retrieval quality separately
    from answer-generation quality — follows Luo et al. (2025, §4.2), allowing
    attribution of errors to either the constrained-decoding component or the
    generation component.

    Parameters
    ----------
    beams          : Ranked list of decoded path strings from the GCR trie.
    gold_answer    : Gold answer string from the evaluation dataset.
    question_family: "next_step" | "counterfactual" | other.
    gold_paths     : ``q["gold_paths"]`` for path_recall / path_precision.
    context_block  : Enriched object context string (used for context_density).

    Returns
    -------
    (prediction_str, scores_dict)
    """
    prediction = " ".join(b for b in beams if b).strip()
    scores = score_answer(prediction, gold_answer, question_family, beams=beams)

    # --- Path recall: did the trie recover the correct reasoning trace? ---
    if gold_paths and beams:
            # Assuming 'beams' is your list of strings
        normalized_beams = [
            [event.replace("Event:", "").replace("_", " ") for event in b.split()] 
            for b in beams
        ]
        #scores["path_recall"] = calculate_path_recall(normalized_beams, gold_paths)
        best_precision, best_recall, best_f1, best_lcs, avg_precision, avg_recall, avg_f1, avg_lcs = best_path_metrics(normalized_beams, gold_paths[0][0])
        scores["path_recall"] = best_recall  # or avg_recall, depending on your preference
        scores["path_precision"] = best_precision  # or avg_precision
        scores["path_f1"] = best_f1  # or avg_f1
        scores["lcs_recall"] = best_lcs  # or avg_lcs
    else:
        scores["path_recall"] = float("nan")

    # # --- Path precision: fraction of beams containing a gold activity ---
    # if gold_paths and beams:
    #     scores["path_precision"] = calculate_path_precision(beams, gold_paths)
    #     print(f"  path_precision = {scores['path_precision']:.4f}")
    # else:
    #     scores["path_precision"] = float("nan")

    # # --- Context density: colon count per 100 chars (attribute-density proxy) ---
    # if context_block:
    #     total_chars = len(context_block)
    #     signal_count = context_block.count(":")
    #     scores["context_density"] = (
    #         signal_count / (total_chars / 100) if total_chars > 0 else 0.0
    #     )

    return prediction, scores

def calculate_efficiency_metrics(file_path):
    with open(file_path, 'r') as f:
        # Assuming the file is a list of JSON objects
        data = json.load(f)

    # Initialize lists for each metric
    trie_times = []
    path_gen_times = []
    enrich_times = []
    total_times = []

    for entry in data:
        trie_times.append(entry.get('trie_build_s', 0))
        path_gen_times.append(entry.get('generation_s', 0))
        enrich_times.append(entry.get('enrich_s', 0))
        total_times.append(entry.get('total_s', 0))

    metrics = {
        "Trie Construction": trie_times,
        "Path Generation": path_gen_times,
        "Context Enrichment": enrich_times,
        "Total Pipeline": total_times
    }

    print(f"{'Metric':<20} | {'Mean (s)':<10} | {'P95 (s)':<10}")
    print("-" * 46)
    
    for name, values in metrics.items():
        mean_val = np.mean(values)
        p95_val = np.percentile(values, 95)
        print(f"{name:<20} | {mean_val:<10.4f} | {p95_val:<10.4f}")



# ===========================================================================
# Main evaluation loop
# ===========================================================================

def run_path_evaluation(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    answers_path = os.path.join(args.out_dir, "path_answers.jsonl")
    done = load_done(answers_path) 

    # ------------------------------------------------------------------ #
    # Load dataset
    # ------------------------------------------------------------------ #
    print(f"Loading dataset: {args.dataset}")
    questions = load_dataset(args.dataset)
    if args.limit:
        questions = questions[: args.limit]
        print(f"  [--limit] Using first {args.limit} instances.")
    print(f"  {len(questions)} instances loaded.\n")

    # ------------------------------------------------------------------ #
    # Load predicted paths
    # ------------------------------------------------------------------ #
    constrained_index:   Dict[str, Dict] = {}
    unconstrained_index: Dict[str, Dict] = {}

    if args.constrained_paths and os.path.exists(args.constrained_paths):
        for rec in load_jsonl(args.constrained_paths):
            constrained_index[rec["instance_id"]] = rec
        print(f"Loaded {len(constrained_index)} constrained path records.")

    if args.unconstrained_paths and os.path.exists(args.unconstrained_paths):
        for rec in load_jsonl(args.unconstrained_paths):
            unconstrained_index[rec["instance_id"]] = rec
        print(f"Loaded {len(unconstrained_index)} unconstrained path records.\n")

    # ------------------------------------------------------------------ #
    # Systems
    # ------------------------------------------------------------------ #
    systems: List[Tuple[str, Dict[str, Dict]]] = []
    if constrained_index:
        systems.append(("gcr_constrained_paths", constrained_index))
    if unconstrained_index:
        systems.append(("gcr_unconstrained_paths", unconstrained_index))

    if not systems:
        print("No path files provided — nothing to evaluate.")
        return

    print(f"Systems to evaluate: {[s for s, _ in systems]}\n")

    # ------------------------------------------------------------------ #
    # Evaluation loop
    # ------------------------------------------------------------------ #
    all_scored = []

    for system, path_index in systems:
        print(f"=== {system.upper()} ===")

        for q in tqdm(questions, desc=f"[{system}]"):
            instance_id     = q["instance_id"]
            if (instance_id, system) in done:
                continue

            gold_answer     = q.get("gold_answer", "")
            gold_paths      = q.get("gold_paths", [])
            question_text   = q["question"]
            question_family = q.get("question_family", "unknown")

            path_rec = path_index.get(instance_id, {})
            beams    = path_rec.get("paths", [])
            ctx      = path_rec.get("context_block", "")

            #print(f"\n[Instance {instance_id}] {question_text}")

            try:
                prediction, metrics = score_paths_directly(
                    beams,
                    gold_answer,
                    question_family,
                    gold_paths=gold_paths,
                    context_block=ctx,
                )
            except Exception as exc:
                print(f"\n  [WARNING] {instance_id} / {system}: {exc}", flush=True)
                prediction = ""
                metrics    = {}

            # Timing fields — no LLM so answer_s = 0
            extra = {
                "answer_s":     0.0,
                "path_total_s": path_rec.get("total_s", 0.0),
                "trie_build_s": path_rec.get("trie_build_s", 0.0),
                "generation_s": path_rec.get("generation_s", 0.0),
                "enrich_s":     path_rec.get("enrich_s", 0.0),
            }

            record = {
                "instance_id":     instance_id,
                "system":          system,
                "question_family": question_family,
                "question":        question_text,
                "gold_answer":     gold_answer,
                "prediction":      prediction,
                **metrics,
                **extra,
            }

            append_record(answers_path, record)
            done.add((instance_id, system))
            all_scored.append(record)

    # ------------------------------------------------------------------ #
    # Aggregate + report
    # ------------------------------------------------------------------ #
    import json
    all_written = load_jsonl(answers_path) if os.path.exists(answers_path) else []
    results = aggregate(all_written)

    print_results_table(results, PATH_METRIC_COLS)

    write_results_table(
        results,
        args.out_dir,
        caption=(
            "Path-retrieval evaluation on 100-instance P2P OCEL benchmark. "
            "EM = exact match against gold answer (surface overlap), "
            "MRR = mean reciprocal rank over GCR beams, "
            "Path-Recall = fraction of gold activity sequences recovered, "
            "Path-Precision = fraction of beams containing a gold activity."
        ),
        label="tab:path_results",
    )
    # rename CSV/LaTeX so they don't collide with eval_answers output
    _rename_outputs(args.out_dir, prefix="path_")

    print(f"\nAll path records → {answers_path}")


def _rename_outputs(out_dir: str, prefix: str) -> None:
    """Rename the generic results_table.* files to <prefix>results_table.*"""
    import shutil
    for ext in ("csv", "tex"):
        src = os.path.join(out_dir, f"results_table.{ext}")
        dst = os.path.join(out_dir, f"{prefix}results_table.{ext}")
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Renamed → {dst}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage 2a: score GCR beam paths directly against gold answers "
            "and gold path sequences (no LLM required)."
        )
    )
    p.add_argument("--dataset", required=True,
                   help="Evaluation sample JSON/JSONL (must contain gold_paths).")
    p.add_argument("--constrained_paths", default=None,
                   help="JSONL from generate_predicted_paths.py (constrained).")
    p.add_argument("--unconstrained_paths", default=None,
                   help="JSONL from generate_predicted_paths.py (unconstrained).")
    p.add_argument("--out_dir", default="results",
                   help="Directory for path_answers.jsonl and path_results_table.*")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N instances (for testing).")
    return p.parse_args()


if __name__ == "__main__":
    run_path_evaluation(parse_args())