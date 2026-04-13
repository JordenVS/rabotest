"""
eval/evaluate.py
----------------
Scores the predictions written by generate_predicted_paths.py.

Metrics
-------
For **next_step** questions (generative, open-ended):
  - Exact Match (EM)       — prediction contains gold_answer as a substring
                              (case-insensitive, normalised).
  - Activity F1            — token-level F1 between predicted and gold activity
                              string (standard QA metric; Rajpurkar et al., 2016).
  - ROUGE-L                — longest-common-subsequence recall/precision/F1
                              (Lin, 2004); standard for generative eval.

For **counterfactual** questions (binary Yes/No):
  - Accuracy               — whether the prediction contains "no" / "yes" in
                              the expected polarity.
  - F1 (binary)            — treating the task as binary classification.

Overall (all families):
  - Mean Reciprocal Rank (MRR) — position of first correct activity mention
                                  across GCR's k beam outputs.
  - Path Validity Rate (PVR)   — fraction of GCR predictions that constitute
                                  a valid walk in the graph (GCR-specific).
  - Latency (mean / p95)       — wall-clock seconds per instance.

Usage
-----
    python -m eval.evaluate \
        --predictions eval/predictions.jsonl \
        --graph       test2.graphml \
        --out-csv     eval/results_table.csv \
        [--out-json   eval/results_full.json]

The CSV is formatted for direct inclusion in a LaTeX table via pandas
``to_latex()``.

References
----------
Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine
    Comprehension of Text. EMNLP.
Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
    ACL Workshop on Text Summarisation Branches Out.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Text normalisation (mirrors SQuAD eval script; Rajpurkar et al., 2016)
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, remove punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _tokenise(text: str) -> List[str]:
    return _normalise(text).split()


# ---------------------------------------------------------------------------
# Token-level F1  (Rajpurkar et al., 2016)
# ---------------------------------------------------------------------------

def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _tokenise(prediction)
    gold_tokens = _tokenise(gold)

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    # Count occurrences (not just set membership)
    from collections import Counter
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    num_same = sum(min(pred_counts[t], gold_counts[t]) for t in common)

    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


# ---------------------------------------------------------------------------
# ROUGE-L  (Lin, 2004)
# ---------------------------------------------------------------------------

def _lcs_length(a: List[str], b: List[str]) -> int:
    """Standard dynamic-programming LCS length."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(prediction: str, gold: str) -> Dict[str, float]:
    pred_tokens = _tokenise(prediction)
    gold_tokens = _tokenise(gold)

    if not pred_tokens or not gold_tokens:
        f = float(pred_tokens == gold_tokens)
        return {"precision": f, "recall": f, "f1": f}

    lcs = _lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall    = lcs / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Exact match (substring, case-insensitive)
# ---------------------------------------------------------------------------

def exact_match(prediction: str, gold: str) -> bool:
    return _normalise(gold) in _normalise(prediction)


# ---------------------------------------------------------------------------
# Counterfactual accuracy  (binary)
# ---------------------------------------------------------------------------

def counterfactual_correct(prediction: str, gold_answer: str) -> bool:
    """
    For counterfactual questions the gold_answer is "Yes" or "No".
    We check whether the first polar word in the prediction matches.
    """
    p = _normalise(prediction)
    gold_polar = _normalise(gold_answer)

    # Look for explicit yes/no in prediction
    has_no  = bool(re.search(r"\bno\b",  p))
    has_yes = bool(re.search(r"\byes\b", p))

    if gold_polar == "no":
        # Correct if "no" present AND "yes" absent (or both absent → conservative)
        return has_no and not has_yes
    else:
        return has_yes and not has_no


# ---------------------------------------------------------------------------
# MRR  (over GCR beam outputs)
# ---------------------------------------------------------------------------

def mrr_from_beams(beams: List[str], gold: str) -> float:
    """
    Mean Reciprocal Rank given a ranked list of beam outputs and a gold string.
    Returns 1/rank of the first beam that contains the gold answer, else 0.
    """
    gold_norm = _normalise(gold)
    for rank, beam in enumerate(beams, start=1):
        if gold_norm in _normalise(beam):
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Path Validity Rate  (GCR-specific)
# ---------------------------------------------------------------------------

def is_valid_walk(path_string: str, graph) -> bool:
    """
    Parse a GCR chain string and verify every consecutive node pair is
    connected by an edge in *graph*.

    Chain format:  "NodeLabel rel NodeLabel rel NodeLabel …"
    Node labels:   "Event:Activity_Name" or "Object:type"

    The node labels must map back to actual graph node IDs.  We do this
    by building a reverse index (label → set of node IDs) once per call.
    For large-scale eval, pass a pre-built reverse index instead.
    """
    import networkx as nx

    tokens = path_string.strip().split()
    # Tokens alternate: NodeLabel, relation, NodeLabel, relation, …
    # Extract node tokens (even indices)
    node_tokens = tokens[::2]

    if len(node_tokens) < 2:
        return False  # trivial/empty path

    # Build reverse-label → node_id mapping from graph
    label_to_ids: Dict[str, List[str]] = defaultdict(list)
    for nid, data in graph.nodes(data=True):
        entity_type = data.get("entity_type", "Node")
        raw = data.get("activity", data.get("object_type", nid))
        label = f"{entity_type}:{raw.replace(' ', '_')}"
        label_to_ids[label].append(nid)

    # Resolve each token to candidate node IDs
    candidates: List[List[str]] = []
    for tok in node_tokens:
        ids = label_to_ids.get(tok, [])
        if not ids:
            return False   # unresolvable label → invalid
        candidates.append(ids)

    # Check that *some* instantiation of candidates forms a valid walk
    # (BFS over the candidate space — bounded because lists are small)
    from itertools import product
    for combo in product(*candidates):
        valid = True
        for i in range(len(combo) - 1):
            u, v = combo[i], combo[i + 1]
            if not graph.has_edge(u, v):
                valid = False
                break
        if valid:
            return True

    return False


# ---------------------------------------------------------------------------
# Score a single record
# ---------------------------------------------------------------------------

def score_record(
    record: Dict[str, Any],
    graph,
    compute_pvr: bool = True,
) -> Dict[str, Any]:
    prediction  = record.get("prediction", "")
    gold        = str(record.get("gold_answer", ""))
    family      = record.get("question_family", "unknown")
    system      = record.get("system", "unknown")
    beams       = record.get("metadata", {}).get("all_beams", [prediction])

    scores: Dict[str, Any] = {
        "sample_id": record["sample_id"],
        "system":    system,
        "family":    family,
    }

    if family == "next_step":
        scores["em"]      = float(exact_match(prediction, gold))
        scores["tok_f1"]  = token_f1(prediction, gold)
        scores["rouge_l"] = rouge_l(prediction, gold)["f1"]
        scores["mrr"]     = mrr_from_beams(beams, gold)

    elif family == "counterfactual":
        scores["cf_acc"] = float(counterfactual_correct(prediction, gold))
        scores["em"]     = scores["cf_acc"]   # alias for unified reporting

    # Path Validity Rate — meaningful mainly for GCR; computed for all
    if compute_pvr and graph is not None:
        scores["pvr"] = float(is_valid_walk(prediction, graph))

    scores["elapsed_s"] = record.get("elapsed_s", float("nan"))

    return scores


# ---------------------------------------------------------------------------
# Aggregate scores into a results table
# ---------------------------------------------------------------------------

def aggregate(
    score_records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Group score records by system and compute mean ± std for each metric.

    Returns
    -------
    Dict mapping system_name → {metric_name: value}
    """
    from collections import defaultdict
    import numpy as np

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for sr in score_records:
        grouped[sr["system"]].append(sr)

    results: Dict[str, Dict[str, float]] = {}

    for system, records in grouped.items():
        def _mean(key):
            vals = [r[key] for r in records if key in r and not np.isnan(r[key])]
            return float(np.mean(vals)) if vals else float("nan")

        def _p95(key):
            vals = [r[key] for r in records if key in r and not np.isnan(r[key])]
            return float(np.percentile(vals, 95)) if vals else float("nan")

        results[system] = {
            "n":             len(records),
            # Generative metrics (next_step subset)
            "em":            _mean("em"),
            "tok_f1":        _mean("tok_f1"),
            "rouge_l":       _mean("rouge_l"),
            "mrr":           _mean("mrr"),
            # Counterfactual
            "cf_acc":        _mean("cf_acc"),
            # Graph fidelity
            "pvr":           _mean("pvr"),
            # Efficiency
            "lat_mean_s":    _mean("elapsed_s"),
            "lat_p95_s":     _p95("elapsed_s"),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score GCR/RAG/GraphRAG predictions against gold annotations."
    )
    parser.add_argument("--predictions", default="eval/predictions.jsonl")
    parser.add_argument("--graph",       default="test2.graphml",
                        help="GraphML file for Path Validity Rate computation.")
    parser.add_argument("--out-csv",     default="eval/results_table.csv")
    parser.add_argument("--out-json",    default=None)
    parser.add_argument("--no-pvr",      action="store_true",
                        help="Skip Path Validity Rate (faster, no graph needed).")
    args = parser.parse_args()

    # ---- Load predictions ----
    records: List[Dict[str, Any]] = []
    with open(args.predictions, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} prediction records.")

    # ---- Load graph ----
    graph = None
    if not args.no_pvr:
        from utils.graph_utils import load_graphml_to_networkx
        print(f"Loading graph: {args.graph}")
        graph = load_graphml_to_networkx(args.graph)

    # ---- Score each record ----
    scored = []
    for rec in records:
        s = score_record(rec, graph, compute_pvr=(not args.no_pvr))
        scored.append(s)

    # ---- Aggregate ----
    results = aggregate(scored)

    # ---- Pretty-print ----
    print("\n========= RESULTS =========")
    METRICS = ["n", "em", "tok_f1", "rouge_l", "mrr", "cf_acc", "pvr",
                "lat_mean_s", "lat_p95_s"]
    header = f"{'System':<12}" + "".join(f"{m:>12}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for sys_name, vals in sorted(results.items()):
        row = f"{sys_name:<12}"
        for m in METRICS:
            v = vals.get(m, float("nan"))
            row += f"{v:>12.4f}" if not isinstance(v, int) else f"{v:>12d}"
        print(row)

    # ---- Save CSV ----
    try:
        import pandas as pd
        df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "system"})
        df.to_csv(args.out_csv, index=False)
        print(f"\nResults CSV → {args.out_csv}")

        # LaTeX table for paper (include in appendix or results section)
        latex_path = args.out_csv.replace(".csv", ".tex")
        df_latex = df.copy()
        # Round floats
        float_cols = [c for c in df_latex.columns if c not in ("system", "n")]
        df_latex[float_cols] = df_latex[float_cols].applymap(
            lambda x: round(x, 4) if isinstance(x, float) else x
        )
        df_latex.to_latex(
            latex_path, index=False, float_format="%.4f",
            caption="Evaluation results on 100-instance P2P OCEL benchmark.",
            label="tab:results",
        )
        print(f"LaTeX table → {latex_path}")
    except ImportError:
        print("pandas not found — CSV/LaTeX output skipped.")

    # ---- Save full JSON ----
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"aggregated": results, "per_instance": scored},
                      f, indent=2, default=str)
        print(f"Full JSON → {args.out_json}")


if __name__ == "__main__":
    main()