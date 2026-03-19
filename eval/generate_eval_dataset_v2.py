"""
generate_eval_dataset_v2.py
============================
Generates a three-type evaluation dataset for GCR on OCEL process graphs.

Question Types
--------------
TYPE 1 — Local Lifecycle Explanation
    Requires semantic interpretation of a retrieved path, not just path retrieval.
    Ground truth: structured facts derivable from the graph (timestamps, object links).

TYPE 2 — Soft Anomaly Detection
    Requires detecting a structural deviation from the majority process variant.
    The majority variant is explicitly provided in the question so all evaluation
    conditions (full, noctx, direct) are tested on the same reasoning task rather
    than on prior LLM knowledge of procurement processes.
    Ground truth: binary (anomalous / normal) + deviation point, computed via
    variant analysis. Constructable without human annotation.

TYPE 3 — Cross-Object Relational Reasoning
    Requires reasoning over multiple connected objects in the local subgraph.
    Ground truth: derivable from graph timestamps and edge structure.

Output Format
-------------
JSONL — one question per line, all three types interleaved and shuffled.
Each record:
{
    "id":               "T1_001",
    "type":             "lifecycle_explanation" | "anomaly_detection" | "cross_object_reasoning",
    "question":         "<natural language question>",
    "topic_entities":   ["<seed node id>", ...],
    "ground_truth": {
        "facts":            { ... },
        "label":            null | true | false,
        "deviation":        null | "<description>",
        "reference_path":   null | "<linearized path string>",
        "path_ground_truth": {
            "seed_node":       "<node id>",
            "expected_nodes":  ["Event:Activity_Name", ...],
            "object_nodes":    ["Object:type", ...]   (Type 3 only)
        }
    },
    "evaluation_notes": "<what a correct answer must contain>"
}

Usage
-----
    python generate_eval_dataset_v2.py \
        --ocel data/ocel2-p2p.json \
        --graphml test2.graphml \
        --out eval_v2 \
        --n1 35 --n2 35 --n3 30 \
        --seed 42

Dependencies
------------
    pip install pm4py networkx pandas tqdm
"""

import argparse
import json
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import pm4py
from tqdm import tqdm


# =============================================================================
# GRAPH UTILITIES
# =============================================================================

def load_graph(graphml_path: str) -> nx.DiGraph:
    G = nx.read_graphml(graphml_path)
    print(f"[graph] Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def node_label(G: nx.DiGraph, n: str) -> str:
    """Human-readable label for a node."""
    data = G.nodes[n]
    if data.get("entity_type") == "Event":
        return f"Event:{data.get('activity', n)}"
    elif data.get("entity_type") == "Object":
        return f"{data.get('object_type', 'Object')}:{n}"
    return n


def linearize_path(G: nx.DiGraph, path: List[Tuple[str, str, str]]) -> str:
    """Convert [(u, rel, v), ...] to readable string."""
    parts = []
    for u, rel, v in path:
        parts += [node_label(G, u), rel, node_label(G, v)]
    return " -> ".join(parts)


def extract_paths(
    G: nx.DiGraph, start: str, max_depth: int = 4
) -> List[List[Tuple[str, str, str]]]:
    """DFS path extraction from start node."""
    paths, stack = [], [(start, [], 0, {start})]
    while stack:
        node, cur, d, visited = stack.pop()
        if d >= max_depth or G.out_degree(node) == 0:
            if cur:
                paths.append(cur)
            continue
        for _, nxt, data in G.out_edges(node, data=True):
            rel = str(data.get("label", "rel")).replace(" ", "_")
            if nxt in visited:
                if cur:
                    paths.append(cur)
                continue
            stack.append((nxt, cur + [(node, rel, nxt)], d + 1, visited | {nxt}))
    return paths


def get_event_objects(G: nx.DiGraph, event_id: str) -> List[Dict]:
    """Return list of {node_id, object_type, qualifier} for objects linked to event."""
    result = []
    for _, obj, data in G.out_edges(event_id, data=True):
        if G.nodes[obj].get("entity_type") == "Object":
            result.append({
                "node_id": obj,
                "object_type": G.nodes[obj].get("object_type", "object"),
                "qualifier": data.get("label", "relates_to"),
            })
    return result


def get_next_events(G: nx.DiGraph, event_id: str) -> List[Dict]:
    """Return list of {node_id, activity, via_object_type} for next events."""
    result = []
    for _, nxt, data in G.out_edges(event_id, data=True):
        if G.nodes[nxt].get("entity_type") == "Event":
            label = data.get("label", "")
            via = label.replace("NEXT_FOR_", "") if "NEXT_FOR_" in label else label
            result.append({
                "node_id": nxt,
                "activity": G.nodes[nxt].get("activity", ""),
                "via_object_type": via,
                "timestamp": G.nodes[nxt].get("timestamp", ""),
            })
    return result


def fmt(activity: str) -> str:
    """Format activity name to match linearize_path output: spaces to underscores."""
    return activity.replace(" ", "_")


# =============================================================================
# VARIANT ANALYSIS (for Type 2 ground truth)
# =============================================================================

def compute_object_variants(
    G: nx.DiGraph, object_type: str
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    For a given object type, compute the activity sequence (variant) for each
    object instance by collecting events linked to it via in-edges.
    Returns:
        majority_variant: List[str] of activity names (most common sequence)
        object_variants:  Dict[object_id -> List[str]]
    """
    object_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("entity_type") == "Object" and d.get("object_type") == object_type
    ]

    object_variants: Dict[str, List[str]] = {}

    for obj in object_nodes:
        linked_events = []
        for u, _, d in G.in_edges(obj, data=True):
            if G.nodes[u].get("entity_type") == "Event":
                ts = G.nodes[u].get("timestamp", "")
                linked_events.append((ts, u))
        linked_events.sort(key=lambda x: x[0])
        variant = [G.nodes[e].get("activity", "") for _, e in linked_events]
        if variant:
            object_variants[obj] = variant

    if not object_variants:
        return [], {}

    variant_counts = Counter(tuple(v) for v in object_variants.values())
    majority_variant = list(variant_counts.most_common(1)[0][0])

    return majority_variant, object_variants


def find_deviation_point(
    observed: List[str], majority: List[str]
) -> Optional[str]:
    """
    Return a human-readable description of where observed deviates from majority,
    or None if it matches.
    """
    if observed == majority:
        return None

    missing = [a for a in majority if a not in observed]
    extra = [a for a in observed if a not in majority]
    order_issues = []

    majority_idx = {a: i for i, a in enumerate(majority)}
    observed_in_majority = [a for a in observed if a in majority_idx]
    for i in range(len(observed_in_majority) - 1):
        if (majority_idx.get(observed_in_majority[i], 0) >
                majority_idx.get(observed_in_majority[i + 1], 0)):
            order_issues.append(
                f"'{observed_in_majority[i + 1]}' appears before "
                f"'{observed_in_majority[i]}'"
            )

    parts = []
    if missing:
        parts.append(f"Missing steps: {missing}")
    if extra:
        parts.append(f"Unexpected steps: {extra}")
    if order_issues:
        parts.append(f"Order violations: {order_issues}")

    return (
        "; ".join(parts) if parts
        else f"Different sequence: observed {observed} vs majority {majority}"
    )


# =============================================================================
# TYPE 1 — LOCAL LIFECYCLE EXPLANATION
# =============================================================================

def make_type1_questions(
    G: nx.DiGraph, n: int, rng: random.Random
) -> List[Dict]:
    """
    Questions requiring semantic interpretation of a lifecycle path.
    Seeds: object nodes with >= 3 linked events.
    """
    candidates = [
        n_id for n_id, data in G.nodes(data=True)
        if data.get("entity_type") == "Object"
        and sum(1 for _ in G.in_edges(n_id)) >= 3
    ]

    if not candidates:
        print("[T1] Warning: no objects with >=3 linked events found.")
        return []

    selected = rng.sample(candidates, min(n, len(candidates)))
    questions = []

    templates = [
        (
            "Describe the full lifecycle of {obj_type} {obj_id}. "
            "What sequence of events occurred, and is there anything unusual "
            "about this sequence?",
            "Answer must correctly state the activity sequence in chronological "
            "order and note any deviations from typical procurement flow."
        ),
        (
            "What happened to {obj_type} {obj_id} after it was first created? "
            "Walk through each step and identify where the most time was spent.",
            "Answer must identify activities in order and correctly attribute the "
            "longest inter-event gap based on timestamps."
        ),
        (
            "{obj_type} {obj_id} is present in the process log. "
            "Based on its event history, does it appear to have followed a "
            "standard process flow?",
            "Answer must reference the actual activities observed and compare "
            "them to a plausible standard P2P flow."
        ),
    ]

    for i, obj_id in enumerate(selected):
        obj_type = G.nodes[obj_id].get("object_type", "object")

        linked = []
        for u, _, _ in G.in_edges(obj_id, data=True):
            if G.nodes[u].get("entity_type") == "Event":
                ts = G.nodes[u].get("timestamp", "")
                act = G.nodes[u].get("activity", "")
                linked.append({"event_id": u, "activity": act, "timestamp": ts})
        linked.sort(key=lambda x: x["timestamp"])

        gaps = []
        for j in range(len(linked) - 1):
            gaps.append({
                "from": linked[j]["activity"],
                "to": linked[j + 1]["activity"],
                "from_ts": linked[j]["timestamp"],
                "to_ts": linked[j + 1]["timestamp"],
            })

        template_q, eval_note = rng.choice(templates)
        question = template_q.format(obj_type=obj_type, obj_id=obj_id)
        activities = [e["activity"] for e in linked]

        questions.append({
            "id": f"T1_{i + 1:03d}",
            "type": "lifecycle_explanation",
            "question": question,
            "topic_entities": [obj_id],
            "ground_truth": {
                "facts": {
                    "object_type": obj_type,
                    "object_id": obj_id,
                    "event_sequence": linked,
                    "inter_event_gaps": gaps,
                    "num_events": len(linked),
                },
                "label": None,
                "deviation": None,
                "reference_path": " -> ".join(activities),
                "path_ground_truth": {
                    "seed_node": obj_id,
                    "expected_nodes": [
                        f"Event:{fmt(e['activity'])}"
                        for e in linked
                    ],
                },
            },
            "evaluation_notes": eval_note,
        })

    return questions


# =============================================================================
# TYPE 2 — SOFT ANOMALY DETECTION
# =============================================================================

def make_type2_questions(
    G: nx.DiGraph,
    ocel_path: str,
    n: int,
    rng: random.Random,
    anomaly_ratio: float = 0.5,
) -> List[Dict]:
    """
    Questions requiring detection of process deviations.
    The majority variant is explicitly stated in every question so that all
    evaluation conditions are tested on sequence comparison reasoning, not on
    prior LLM knowledge of procurement processes.
    """
    object_types = list({
        d.get("object_type")
        for _, d in G.nodes(data=True)
        if d.get("entity_type") == "Object" and d.get("object_type")
    })

    # Select the object type with the richest variant diversity
    best_type = None
    best_majority: List[str] = []
    best_variants: Dict[str, List[str]] = {}
    best_diversity = 0.0

    for ot in object_types:
        majority, variants = compute_object_variants(G, ot)
        if not majority or len(variants) < 5:
            continue
        n_anomalous = sum(1 for v in variants.values() if list(v) != majority)
        diversity = n_anomalous / len(variants)
        if diversity > best_diversity:
            best_diversity = diversity
            best_type = ot
            best_majority = majority
            best_variants = variants

    if best_type is None:
        print("[T2] Warning: could not find object type with sufficient variant diversity.")
        return []

    print(
        f"[T2] Using object type '{best_type}' "
        f"(majority variant length={len(best_majority)}, "
        f"n_objects={len(best_variants)}, anomaly_rate={best_diversity:.2%})"
    )

    normal_pool = [
        (obj_id, v) for obj_id, v in best_variants.items()
        if list(v) == best_majority
    ]
    anomaly_pool = [
        (obj_id, v) for obj_id, v in best_variants.items()
        if list(v) != best_majority
    ]

    n_anomalous = int(n * anomaly_ratio)
    n_normal = n - n_anomalous

    selected_anomalous = rng.sample(anomaly_pool, min(n_anomalous, len(anomaly_pool)))
    selected_normal = rng.sample(normal_pool, min(n_normal, len(normal_pool)))

    all_selected = (
        [(obj_id, v, True) for obj_id, v in selected_anomalous] +
        [(obj_id, v, False) for obj_id, v in selected_normal]
    )
    rng.shuffle(all_selected)

    # Majority variant is explicitly provided in every template so the model
    # is tested on comparison reasoning, not on procurement domain knowledge.
    templates_anomalous = [
        (
            "The standard process flow for {obj_type} is: {majority_steps}. "
            "However, {obj_type} {obj_id} went through these steps instead: "
            "{observed_steps}. Identify what is missing or out of order.",
            "Answer must correctly identify the specific missing or reordered "
            "steps by comparing observed against the provided standard flow."
        ),
        (
            "Given that the expected process for {obj_type} is: {majority_steps}, "
            "does the following sequence for {obj_id} conform to it: "
            "{observed_steps}? If not, describe the deviation precisely.",
            "Answer must name the specific deviation point relative to the "
            "provided standard flow."
        ),
    ]

    templates_normal = [
        (
            "The standard process flow for {obj_type} is: {majority_steps}. "
            "{obj_type} {obj_id} went through these steps: {observed_steps}. "
            "Does this conform to the standard flow?",
            "Answer must correctly confirm conformance — sequence matches the "
            "provided standard and no false deviations should be flagged."
        ),
        (
            "Given that the expected process for {obj_type} is: {majority_steps}, "
            "does the following sequence for {obj_id} conform: {observed_steps}?",
            "Answer must correctly confirm the sequence is normal."
        ),
    ]

    questions = []
    for i, (obj_id, variant, is_anomalous) in enumerate(all_selected):
        observed_steps = " -> ".join(variant)
        majority_steps = " -> ".join(best_majority)
        deviation = (
            find_deviation_point(list(variant), best_majority)
            if is_anomalous else None
        )

        templates = templates_anomalous if is_anomalous else templates_normal
        template_q, eval_note = rng.choice(templates)

        question = template_q.format(
            obj_type=best_type,
            obj_id=obj_id,
            observed_steps=observed_steps,
            majority_steps=majority_steps,
        )

        questions.append({
            "id": f"T2_{i + 1:03d}",
            "type": "anomaly_detection",
            "question": question,
            "topic_entities": [obj_id],
            "ground_truth": {
                "facts": {
                    "object_type": best_type,
                    "object_id": obj_id,
                    "observed_variant": list(variant),
                    "majority_variant": best_majority,
                },
                "label": is_anomalous,
                "deviation": deviation,
                "reference_path": " -> ".join(best_majority),
                "path_ground_truth": {
                    "seed_node": obj_id,
                    "expected_nodes": [
                        f"Event:{fmt(a)}"
                        for a in best_majority
                    ],
                },
            },
            "evaluation_notes": eval_note,
        })

    return questions


# =============================================================================
# TYPE 3 — CROSS-OBJECT RELATIONAL REASONING
# =============================================================================

def make_type3_questions(
    G: nx.DiGraph, n: int, rng: random.Random
) -> List[Dict]:
    """
    Questions requiring reasoning over multiple connected objects in the subgraph.
    Seeds: events with >= 2 distinct object types linked.
    """
    candidates = []
    for evt_id, data in G.nodes(data=True):
        if data.get("entity_type") != "Event":
            continue
        linked_objs = get_event_objects(G, evt_id)
        obj_types = {o["object_type"] for o in linked_objs}
        if len(obj_types) >= 2:
            candidates.append((evt_id, linked_objs))

    if not candidates:
        print("[T3] Warning: no events with >=2 distinct object types found.")
        return []

    selected = rng.sample(candidates, min(n, len(candidates)))
    questions = []

    for i, (evt_id, linked_objs) in enumerate(selected):
        activity = G.nodes[evt_id].get("activity", "")
        timestamp = G.nodes[evt_id].get("timestamp", "")
        next_evts = get_next_events(G, evt_id)

        obj_context = []
        for obj in linked_objs:
            oid = obj["node_id"]
            otype = obj["object_type"]
            obj_events = []
            for u, _, _ in G.in_edges(oid, data=True):
                if G.nodes[u].get("entity_type") == "Event":
                    ts = G.nodes[u].get("timestamp", "")
                    act = G.nodes[u].get("activity", "")
                    obj_events.append({"event_id": u, "activity": act, "timestamp": ts})
            obj_events.sort(key=lambda x: x["timestamp"])
            obj_context.append({
                "object_id": oid,
                "object_type": otype,
                "qualifier": obj["qualifier"],
                "event_sequence": obj_events,
            })

        if len(obj_context) >= 2:
            obj_a = obj_context[0]
            obj_b = obj_context[1]

            templates = [
                (
                    "Event {evt_id} ({activity}) is linked to both "
                    "{obj_a_type} {obj_a_id} and {obj_b_type} {obj_b_id}. "
                    "Which of these two objects had its own process completed "
                    "first, and what does this suggest about the sequencing of "
                    "this procurement case?",
                    "Answer must correctly identify which object had fewer/earlier "
                    "subsequent events based on their timestamps, and draw a "
                    "reasonable process inference."
                ),
                (
                    "At the time of event {evt_id} ({activity}), "
                    "{obj_a_type} {obj_a_id} and {obj_b_type} {obj_b_id} were "
                    "both involved. Describe what each object's history looks "
                    "like and whether their lifecycles appear to be in sync.",
                    "Answer must accurately describe the event sequences for both "
                    "objects and correctly assess whether their timelines are aligned."
                ),
                (
                    "Given that event {evt_id} involves {obj_a_type} {obj_a_id} "
                    "and {obj_b_type} {obj_b_id}, what are the next expected steps "
                    "for each of these objects based on their position in the process?",
                    "Answer must correctly state the next events for each object "
                    "as derivable from the graph, without fabricating activities."
                ),
            ]

            template_q, eval_note = rng.choice(templates)
            question = template_q.format(
                evt_id=evt_id,
                activity=activity,
                obj_a_type=obj_a["object_type"],
                obj_a_id=obj_a["object_id"],
                obj_b_type=obj_b["object_type"],
                obj_b_id=obj_b["object_id"],
            )

            questions.append({
                "id": f"T3_{i + 1:03d}",
                "type": "cross_object_reasoning",
                "question": question,
                "topic_entities": [evt_id, obj_a["object_id"], obj_b["object_id"]],
                "ground_truth": {
                    "facts": {
                        "event_id": evt_id,
                        "activity": activity,
                        "timestamp": timestamp,
                        "next_events": next_evts,
                        "object_a": obj_a,
                        "object_b": obj_b,
                    },
                    "label": None,
                    "deviation": None,
                    "reference_path": None,
                    "path_ground_truth": {
                        "seed_node": evt_id,
                        "expected_nodes": [
                            f"Event:{fmt(e['activity'])}"
                            for e in obj_a["event_sequence"] + obj_b["event_sequence"]
                        ],
                        "object_nodes": [
                            f"Object:{obj_a['object_type']}",
                            f"Object:{obj_b['object_type']}",
                        ],
                    },
                },
                "evaluation_notes": eval_note,
            })

    return questions


# =============================================================================
# DATASET STATISTICS SUMMARY
# =============================================================================

def print_stats(questions: List[Dict]):
    type_counts = Counter(q["type"] for q in questions)
    print("\n========== DATASET STATISTICS ==========")
    print(f"Total questions: {len(questions)}")
    for t, c in type_counts.items():
        print(f"  {t}: {c}")

    t2 = [q for q in questions if q["type"] == "anomaly_detection"]
    if t2:
        n_pos = sum(1 for q in t2 if q["ground_truth"]["label"] is True)
        n_neg = len(t2) - n_pos
        print(f"\n  Type 2 label balance: {n_pos} anomalous / {n_neg} normal")
    print("=========================================\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation dataset v2.")
    parser.add_argument("--ocel", required=True, help="Path to ocel2-p2p.json")
    parser.add_argument("--graphml", required=True, help="Path to test2.graphml")
    parser.add_argument("--out", default="eval_v2", help="Output prefix (no extension)")
    parser.add_argument("--n1", type=int, default=35, help="Number of Type 1 questions")
    parser.add_argument("--n2", type=int, default=35, help="Number of Type 2 questions")
    parser.add_argument("--n3", type=int, default=30, help="Number of Type 3 questions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    G = load_graph(args.graphml)

    print("\n--- Generating Type 1: Lifecycle Explanation ---")
    t1 = make_type1_questions(G, args.n1, rng)
    print(f"    Generated {len(t1)} questions.")

    print("\n--- Generating Type 2: Soft Anomaly Detection ---")
    t2 = make_type2_questions(G, args.ocel, args.n2, rng)
    print(f"    Generated {len(t2)} questions.")

    print("\n--- Generating Type 3: Cross-Object Relational Reasoning ---")
    t3 = make_type3_questions(G, args.n3, rng)
    print(f"    Generated {len(t3)} questions.")

    all_questions = t1 + t2 + t3
    rng.shuffle(all_questions)

    print_stats(all_questions)

    # Write combined JSONL
    out_combined = f"eval/data/{args.out}_combined.jsonl"
    with open(out_combined, "w") as f:
        for q in all_questions:
            f.write(json.dumps(q) + "\n")
    print(f"Written: {out_combined}")

    # Write per-type JSONL
    for type_name, qs in [
        ("lifecycle_explanation", t1),
        ("anomaly_detection", t2),
        ("cross_object_reasoning", t3),
    ]:
        out_path = f"eval/data/{args.out}_{type_name}.jsonl"
        with open(out_path, "w") as f:
            for q in qs:
                f.write(json.dumps(q) + "\n")
        print(f"Written: {out_path}")

    # Write human-readable summary JSON for manual validation
    out_summary = f"eval/data/{args.out}_summary.json"
    summary = {
        "total": len(all_questions),
        "by_type": {
            "lifecycle_explanation": len(t1),
            "anomaly_detection": len(t2),
            "cross_object_reasoning": len(t3),
        },
        "sample_questions": {
            "T1": t1[0] if t1 else None,
            "T2_anomalous": next(
                (q for q in t2 if q["ground_truth"]["label"]), None
            ),
            "T2_normal": next(
                (q for q in t2 if not q["ground_truth"]["label"]), None
            ),
            "T3": t3[0] if t3 else None,
        },
    }
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Written: {out_summary}")


if __name__ == "__main__":
    main()