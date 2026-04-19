"""
eval/metrics.py
===============
Shared metric functions and I/O helpers for the GCR evaluation pipeline.

Imported by both eval_paths.py (path-retrieval evaluation) and
eval_answers.py (answer-generation evaluation).  Keeping metrics in one
place guarantees that both evaluation scripts score predictions identically,
which is a basic reproducibility requirement for published results.

Metrics implemented
-------------------
exact_match       : Rajpurkar et al. (2016) — normalised substring match.
token_f1          : Rajpurkar et al. (2016) — token-overlap F1.
rouge_l           : Lin (2004) — longest-common-subsequence F1.
mrr               : Mean Reciprocal Rank over a ranked beam list.
counterfactual_acc: Binary yes/no polarity accuracy.
path_recall       : Fraction of gold activity sequences recovered in beams.
path_precision    : Fraction of beams that contain at least one gold activity.

References
----------
Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016).
    SQuAD: 100,000+ questions for machine comprehension of text. EMNLP.
Lin, C.-Y. (2004).
    ROUGE: A package for automatic evaluation of summaries.
    ACL Workshop on Text Summarisation Branches Out.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np


# ===========================================================================
# I/O helpers
# ===========================================================================

def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_dataset(path: str) -> List[Dict]:
    """Accepts a JSON array or JSONL file."""
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(l) for l in content.splitlines() if l.strip()]


def load_done(path: str) -> set:
    """Return (instance_id, system) pairs already written — for resume support."""
    done = set()
    if not __import__("os").path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["instance_id"], r["system"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def append_record(path: str, record: Dict) -> None:
    """Append one JSON record to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ===========================================================================
# Text normalisation
# ===========================================================================

def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _tokens(text: str) -> List[str]:
    return normalise(text).split()


def _create_event(event_str: str) -> str:
    """
    Normalise an activity name into the canonical event-node format used in
    the serialised path strings: "event:<activity_with_underscores>".
    """
    s = normalise(event_str).replace(" ", "_")
    return s if s.startswith("event:") else "event:" + s


# ===========================================================================
# Core metrics  (Rajpurkar et al., 2016; Lin, 2004)
# ===========================================================================

def exact_match(prediction: str, gold: str) -> float:
    """Normalised substring exact match."""
    return float(normalise(gold) in normalise(prediction))


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 (Rajpurkar et al., 2016)."""
    pred_toks = _tokens(prediction)
    gold_toks = _tokens(gold)
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks)
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r)


def _lcs(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (
                dp[i - 1][j - 1] + 1
                if a[i - 1] == b[j - 1]
                else max(dp[i - 1][j], dp[i][j - 1])
            )
    return dp[m][n]


def rouge_l(prediction: str, gold: str) -> float:
    """ROUGE-L F1 (Lin, 2004)."""
    p_toks = _tokens(prediction)
    g_toks = _tokens(gold)
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    lcs = _lcs(p_toks, g_toks)
    prec = lcs / len(p_toks)
    rec  = lcs / len(g_toks)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def mrr(beams: List[str], gold: str) -> float:
    """Mean Reciprocal Rank over a ranked beam list."""
    gold_norm = normalise(gold)
    for rank, beam in enumerate(beams, start=1):
        if gold_norm in normalise(beam):
            return 1.0 / rank
    return 0.0


def counterfactual_acc(prediction: str, gold_answer: str) -> float:
    """
    Binary accuracy for counterfactual questions.
    gold_answer must be "Yes" or "No"; the first polar word in prediction is used.
    """
    p = normalise(prediction)
    gold_polar = normalise(gold_answer)
    has_no  = bool(re.search(r"\bno\b",  p))
    has_yes = bool(re.search(r"\byes\b", p))
    if gold_polar == "no":
        return float(has_no and not has_yes)
    return float(has_yes and not has_no)


def score_answer(
    prediction: str,
    gold_answer: str,
    question_family: str,
    beams: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute all applicable metrics for one (prediction, gold) pair."""
    scores: Dict[str, float] = {}
    if question_family == "next_step":
        scores["em"]      = exact_match(prediction, gold_answer)
        scores["tok_f1"]  = token_f1(prediction, gold_answer)
        scores["rouge_l"] = rouge_l(prediction, gold_answer)
        scores["mrr"]     = mrr(beams or [prediction], gold_answer)
    elif question_family == "counterfactual":
        scores["cf_acc"] = counterfactual_acc(prediction, gold_answer)
        scores["em"]     = scores["cf_acc"]
    else:
        scores["em"] = exact_match(prediction, gold_answer)
    return scores


# ===========================================================================
# Path-specific metrics
# ===========================================================================

def calculate_path_recall(
    predicted_beams: List[List[str]],
    gold_paths: List[List[List[str]]],
) -> float:
    """
    Measures what fraction of the gold activity sequences are recoverable
    from the predicted beams.

    Gold format: a list of path-sets, where each path-set is a list of
    activity sequences (List[str]).  A gold path-set is considered "found"
    if *any* of its sequences has all activities present in the combined
    beam string (order-insensitive heuristic).

    Parameters
    ----------
    predicted_beams : Ranked list of decoded path strings.
    gold_paths      : ``q["gold_paths"]`` from the evaluation dataset.

    Returns
    -------
    float in [0, 1].
    """
    if not gold_paths:
        return 1.0
    combined = " ".join(predicted_beams).lower()
    found = sum(
        1
        for path_set in gold_paths
        for sequence in path_set
        if all(_create_event(act) in combined for act in sequence)
    )
    return found / len(gold_paths)


def calculate_path_precision(
    predicted_beams: List[str],
    gold_paths: List[List[List[str]]],
) -> float:
    """
    Fraction of predicted beams that contain at least one gold activity.

    A beam is considered a "hit" if any activity name from the gold paths
    appears (case-insensitive substring) in that beam string.

    Parameters
    ----------
    predicted_beams : Ranked list of decoded path strings.
    gold_paths      : ``q["gold_paths"]`` from the evaluation dataset.

    Returns
    -------
    float in [0, 1], or NaN if beams is empty.
    """
    if not predicted_beams or not gold_paths:
        return float("nan")
    gold_acts = {
        act.lower()
        for path_set in gold_paths
        for seq in path_set
        for act in seq
    }
    hits = sum(
        1 for b in predicted_beams if any(ga in b.lower() for ga in gold_acts)
    )
    return hits / len(predicted_beams)

def calculate_lcs_similarity(predicted_path, actual_path):
    # Standard Dynamic Programming approach for LCS
    n, m = len(predicted_path), len(actual_path)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if predicted_path[i-1] == actual_path[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[n][m]

    #lcs_precision = lcs_length / len(predicted_path) if len(predicted_path) > 0 else 0
    lcs_recall = lcs_length / len(actual_path) if len(actual_path) > 0 else 0
    #lcs_f1 = 2 * lcs_precision * lcs_recall / (lcs_precision + lcs_recall) if (lcs_precision + lcs_recall) > 0 else 0
    
    return lcs_recall # This is usually the most useful "Path Accuracy" score

def get_transitions(path):
    # Turns [A, B, C] into {(A, B), (B, C)}
    return set(zip(path, path[1:]))

def calculate_path_metrics(predicted_path, ground_truth):
    pred_edges = get_transitions(predicted_path)
    actual_edges = get_transitions(ground_truth)
    
    tp = len(pred_edges.intersection(actual_edges))
    fp = len(pred_edges - actual_edges)
    fn = len(actual_edges - pred_edges)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def best_path_metrics(normalized_beams, gold_paths):
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_lcs = []
    for beam in normalized_beams:
        p, r, f = calculate_path_metrics(beam, gold_paths) # Using transition function
        all_precisions.append(p)
        all_recalls.append(r)
        all_f1s.append(f)
        lcs = calculate_lcs_similarity(beam, gold_paths)
        all_lcs.append(lcs)

    # 1. Best-of-N (The "Top" performance)
    best_precision = max(all_precisions)
    best_recall = max(all_recalls)
    best_f1 = max(all_f1s)
    best_lcs = max(all_lcs)

    # 2. Average (The "Reliability" performance)
    avg_precision = sum(all_precisions) / len(all_precisions)
    avg_recall = sum(all_recalls) / len(all_recalls)
    avg_f1 = sum(all_f1s) / len(all_f1s)
    avg_lcs = sum(all_lcs) / len(all_lcs)

    return best_precision, best_recall, best_f1, best_lcs, avg_precision, avg_recall, avg_f1, avg_lcs

# ===========================================================================
# Aggregation
# ===========================================================================

def aggregate(scored: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Group per-instance score dicts by system and return mean metrics.

    Handles NaN values gracefully — only non-NaN entries contribute to means.
    """
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for s in scored:
        grouped[s["system"]].append(s)

    results = {}
    for system, recs in grouped.items():

        def _mean(key: str) -> float:
            vals = [
                r[key] for r in recs
                if key in r
                and r[key] is not None
                and not (isinstance(r[key], float) and np.isnan(r[key]))
            ]
            return float(np.mean(vals)) if vals else float("nan")

        def _p95(key: str) -> float:
            vals = [
                r[key] for r in recs
                if key in r
                and r[key] is not None
                and not (isinstance(r[key], float) and np.isnan(r[key]))
            ]
            return float(np.percentile(vals, 95)) if vals else float("nan")

        # results[system] = {
        #     "n":              len(recs),
        #     "em":             _mean("em"),
        #     "tok_f1":         _mean("tok_f1"),
        #     "rouge_l":        _mean("rouge_l"),
        #     "mrr":            _mean("mrr"),
        #     "path_recall":    _mean("path_recall"),
        #     "path_precision": _mean("path_precision"),
        #     "cf_acc":         _mean("cf_acc"),
        #     "lat_mean_s":     _mean("answer_s"),
        #     "lat_p95_s":      _p95("answer_s"),
        # }
        results[system] = {
            "n":              len(recs),
            "em":             _mean("em"),
            "path_recall":             _mean("path_recall"),
            "path_precision":         _mean("path_precision"),
            "path_f1":        _mean("path_f1"),
            "lcs_recall":            _mean("lcs_recall"),
            "lat_mean_s":     _mean("answer_s"),
            "lat_p95_s":      _p95("answer_s"),
            "prompt_tokens_mean":     _mean("prompt_tokens_answer"),
            "completion_tokens_mean": _mean("completion_tokens"),
        }
    return results


# ===========================================================================
# Results reporting  (shared table printer + CSV/LaTeX writer)
# ===========================================================================

# PATH_METRIC_COLS    = ["n", "em", "tok_f1", "rouge_l", "mrr",
#                        "path_recall", "path_precision"]
PATH_METRIC_COLS    = ["n", "path_recall", "path_precision", "path_f1", "lcs_recall"]
ANSWER_METRIC_COLS  = ["em", "prompt_tokens_mean", "completion_tokens_mean",
    "lat_mean_s", "lat_p95_s"]
ALL_METRIC_COLS     = ["n", "em", "tok_f1", "rouge_l", "mrr",
                       "path_recall", "path_precision", "cf_acc",
                       "lat_mean_s", "lat_p95_s"]


def print_results_table(results: Dict[str, Dict], metric_cols: List[str]) -> None:
    """Pretty-print an aggregated results dict to stdout."""
    print("\n========= RESULTS =========")
    header = f"{'System':<28}" + "".join(f"{m:>14}" for m in metric_cols)
    print(header)
    print("-" * len(header))
    for sys_name, vals in sorted(results.items()):
        row = f"{sys_name:<28}"
        for m in metric_cols:
            v = vals.get(m, float("nan"))
            row += f"{v:>14d}" if isinstance(v, int) else f"{v:>14.4f}"
        print(row)


def write_results_table(
    results: Dict[str, Dict],
    out_dir: str,
    caption: str = "",
    label: str = "tab:results",
) -> None:
    """Write results to CSV and LaTeX files in *out_dir*."""
    import os
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available — skipping CSV/LaTeX output.")
        return

    df = (
        pd.DataFrame(results).T
        .reset_index()
        .rename(columns={"index": "system"})
    )

    csv_path = os.path.join(out_dir, "results_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV  → {csv_path}")

    tex_path = os.path.join(out_dir, "results_table.tex")
    float_cols = [c for c in df.columns if c not in ("system", "n")]
    df[float_cols] = df[float_cols].map(
        lambda x: round(float(x), 4) if pd.notnull(x) else x
    )
    df.to_latex(
        tex_path,
        index=False,
        float_format="%.4f",
        caption=caption,
        label=label,
    )
    print(f"LaTeX → {tex_path}")