from __future__ import annotations
import random
import uuid
from collections import defaultdict
import argparse
import json
from typing import Any, Dict, List
 
import pm4py
import networkx as nx
 
# Project-internal imports
from utils.graph_utils import load_graphml_to_networkx

QUESTION_TEMPLATES = {
    "next_step": [
        "What happened next for {obj} after {act}?",
        "Which activity followed {act} for {obj}?"
    ],
    "why": [
        "Why did {act2} occur for {obj}?",
        "Why was {obj} processed with {act2}?"
    ],
    "counterfactual": [
        "Could {act2} have occurred immediately after {act1} for {obj}?",
        "Was it possible for {act2} to happen before {act1} for {obj}?"
    ]
}

def extract_object_lifecycles(ocel):
    """
    Returns:
      lifecycles: dict
        oid -> {
          "object_type": str,
          "activities": [activity_1, activity_2, ...]
        }
    """
    events_df = ocel.events.sort_values("ocel:timestamp")
    objects_df = ocel.objects
    relations_df = ocel.relations

    lifecycles = {}

    grouped = relations_df.groupby("ocel:oid")

    for oid, group in grouped:
        event_ids = group["ocel:eid"].unique()
        subset = events_df[events_df["ocel:eid"].isin(event_ids)]

        activities = subset["ocel:activity"].tolist()
        if len(activities) < 2:
            continue

        obj_type = objects_df.loc[
            objects_df["ocel:oid"] == oid, "ocel:type"
        ].values

        lifecycles[oid] = {
            "object_type": obj_type[0] if len(obj_type) else "object",
            "activities": activities
        }

    return lifecycles

def generate_positive_examples(lifecycles, max_per_object=3):
    examples = []

    for oid, data in lifecycles.items():
        acts = data["activities"]

        for i in range(min(len(acts) - 1, max_per_object)):
            act1, act2 = acts[i], acts[i + 1]

            q_template = random.choice(QUESTION_TEMPLATES["next_step"])
            q = q_template.format(obj=oid, act=act1)

            examples.append({
                "anchor_oid": oid,
                "object_type": data["object_type"],
                "question_family": "next_step",
                "question": q,
                "gold_paths": [acts],
                "gold_answer": act2,
                "behaviorally_valid": True
            })

    return examples

def generate_counterfactual_examples(lifecycles, all_activities, max_per_object=2):
    examples = []

    for oid, data in lifecycles.items():
        acts = data["activities"]

        invalid_candidates = list(set(all_activities) - set(acts))
        if not invalid_candidates:
            continue

        for _ in range(min(max_per_object, len(acts) - 1)):
            act1 = random.choice(acts[:-1])
            act2 = random.choice(invalid_candidates)

            q_template = random.choice(QUESTION_TEMPLATES["counterfactual"])
            q = q_template.format(
                obj=oid,
                act1=act1,
                act2=act2
            )

            examples.append({
                "anchor_oid": oid,
                "object_type": data["object_type"],
                "question_family": "counterfactual",
                "question": q,
                "gold_paths": [acts],
                "gold_answer": "No",
                "behaviorally_valid": False
            })

    return examples

def extract_context_snapshot(G_context, oid, max_depth=1):
    """
    Extract a small ego-graph around the anchor object.
    """
    nodes = set([oid])
    frontier = set([oid])

    for _ in range(max_depth):
        next_frontier = set()
        for n in frontier:
            neighbors = G_context.neighbors(n)
            next_frontier.update(neighbors)
        nodes.update(next_frontier)
        frontier = next_frontier

    subgraph = G_context.subgraph(nodes)

    return {
        "nodes": [
            {"id": n, **subgraph.nodes[n]}
            for n in subgraph.nodes
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                **data
            }
            for u, v, data in subgraph.edges(data=True)
        ]
    }

def build_evaluation_dataset(
    ocel,
    G_context,
    seed=42
):
    random.seed(seed)

    lifecycles = extract_object_lifecycles(ocel)
    all_activities = list(set(ocel.events["ocel:activity"]))

    positives = generate_positive_examples(lifecycles)
    negatives = generate_counterfactual_examples(lifecycles, all_activities)

    dataset = []
    counter = 0

    for ex in positives + negatives:
        instance_id = f"{ex['anchor_oid']}_{counter}"
        counter += 1

        context = extract_context_snapshot(
            G_context,
            ex["anchor_oid"]
        )

        dataset.append({
            "instance_id": instance_id,
            "anchor_object": {
                "oid": ex["anchor_oid"],
                "type": ex["object_type"]
            },
            "question_family": ex["question_family"],
            "question": ex["question"],
            "gold_paths": [ex["gold_paths"]],
            "gold_answer": ex["gold_answer"],
            "behaviorally_valid": ex["behaviorally_valid"],
            "context_snapshot": context
        })

    return dataset

"""
eval/sample_dataset.py
----------------------
Stratified sampler for the OCEL process-mining evaluation framework.
 
Takes the full dataset produced by build_evaluation_dataset() and draws a
fixed-size stratified sample that is balanced across:
  - question_family  (next_step, counterfactual)
  - object_type      (purchase_order, goods_receipt, …)
 
The sample is serialised as a JSON file so every downstream script (path
generation, scoring) operates on the *same* fixed question set — a
prerequisite for reproducible academic comparison.
 
Usage
-----
    python -m eval.sample_dataset \
        --ocel  data/ocel2-p2p.json \
        --graph graphs/context_graph.graphml \
        --out   eval/sampled_100.json \
        --n     100 \
        --seed  42

            python -m eval.sample_dataset --ocel  data/ocel2-p2p.json --graph graphs/context_graph.graphml --out eval/sampled_100.json --n 100 --seed  42
"""
 
 
# ---------------------------------------------------------------------------
# Stratified sampler
# ---------------------------------------------------------------------------
 
def stratified_sample(
    dataset: List[Dict[str, Any]],
    n: int = 100,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Draw *n* instances from *dataset* with stratification over the
    cross-product of (question_family × object_type).
 
    If a stratum has fewer items than its fair share, all items in that
    stratum are kept and the deficit is redistributed proportionally to
    larger strata — ensuring exactly *n* items are returned whenever
    ``len(dataset) >= n``.
 
    Parameters
    ----------
    dataset : full dataset from build_evaluation_dataset()
    n       : target sample size
    seed    : random seed for reproducibility (reported in paper)
 
    Returns
    -------
    List[Dict] — sampled instances, each augmented with a ``sample_id``
    field (zero-padded index within the sample, e.g. "Q000"…"Q099").
    """
    rng = random.Random(seed)
 
    # ---- group by stratum key ----
    strata: Dict[str, List[Dict]] = defaultdict(list)
    for item in dataset:
        family = item.get("question_family", "unknown")
        obj_type = item.get("anchor_object", {}).get("type", "unknown")
        key = f"{family}__{obj_type}"
        strata[key].append(item)
 
    # Shuffle each stratum independently (reproducibly)
    for key in strata:
        rng.shuffle(strata[key])
 
    num_strata = len(strata)
    base_per_stratum = n // num_strata
    remainder = n % num_strata
 
    # Sort strata keys for determinism
    sorted_keys = sorted(strata.keys())
 
    selected: List[Dict] = []
    shortfall = 0
 
    # First pass: collect min(base_per_stratum, stratum_size) per stratum
    quotas: Dict[str, int] = {}
    for i, key in enumerate(sorted_keys):
        quota = base_per_stratum + (1 if i < remainder else 0)
        actual = min(quota, len(strata[key]))
        quotas[key] = actual
        shortfall += quota - actual
 
    # Second pass: redistribute shortfall to strata with spare capacity
    if shortfall > 0:
        for key in sorted_keys:
            spare = len(strata[key]) - quotas[key]
            take = min(spare, shortfall)
            quotas[key] += take
            shortfall -= take
            if shortfall == 0:
                break
 
    # Collect according to quotas
    for key in sorted_keys:
        selected.extend(strata[key][:quotas[key]])
 
    # Shuffle the final sample so families are interleaved
    rng.shuffle(selected)
 
    # Assign stable sample IDs
    for i, item in enumerate(selected):
        item["sample_id"] = f"Q{i:03d}"
 
    return selected
 
 
# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(
        description="Build and stratified-sample the GCR/RAG evaluation dataset."
    )
    parser.add_argument("--ocel",  default="data/ocel2-p2p.json",
                        help="Path to the OCEL 2.0 JSON log.")
    parser.add_argument("--graph", default="test2.graphml",
                        help="Path to the GraphML process graph.")
    parser.add_argument("--out",   default="eval/data/sampled_100.json",
                        help="Output path for the sampled dataset JSON.")
    parser.add_argument("--n",     type=int, default=100,
                        help="Number of evaluation instances (default: 100).")
    parser.add_argument("--seed",  type=int, default=42,
                        help="Random seed (default: 42, reported in paper).")
    args = parser.parse_args()
 
    # 1. Load OCEL log and graph
    print(f"Loading OCEL log: {args.ocel}")
    ocel = pm4py.read_ocel2(args.ocel)
 
    print(f"Loading process graph: {args.graph}")
    G = load_graphml_to_networkx(args.graph)
 
    # 2. Build the full dataset
    print("Building full evaluation dataset…")
    full_dataset = build_evaluation_dataset(ocel, G, seed=args.seed)
    print(f"Full dataset size: {len(full_dataset)} instances")
 
    # 3. Stratified sample
    print(f"Sampling {args.n} instances (seed={args.seed})…")
    sample = stratified_sample(full_dataset, n=args.n, seed=args.seed)
    print(f"Sampled {len(sample)} instances")
 
    # 4. Report stratum breakdown (useful for paper's Table 1)
    family_counts: Dict[str, int] = defaultdict(int)
    type_counts:   Dict[str, int] = defaultdict(int)
    for item in sample:
        family_counts[item["question_family"]] += 1
        type_counts[item["anchor_object"]["type"]] += 1
 
    print("\n=== Stratum breakdown ===")
    print("By question family:")
    for k, v in sorted(family_counts.items()):
        print(f"  {k:30s} {v:4d}")
    print("By object type:")
    for k, v in sorted(type_counts.items()):
        print(f"  {k:30s} {v:4d}")
 
    # 5. Serialise
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, default=str)
    print(f"\nSaved {len(sample)} instances → {args.out}")
 
 
if __name__ == "__main__":
    main()