"""
eval/metrics.py
---------------
All evaluation metric functions for GCR-on-OCEL experiments.

Metrics are grouped into four categories, mirroring the evaluation
dimensions described in the thesis methods chapter:

  1. Correctness  — Hit and F1 (adapted from Luo et al., 2024)
  2. Faithfulness — constraint compliance and hallucination rate
  3. Process-specific — temporal validity and lifecycle coverage
  4. Path quality — token-overlap F1 between generated and ground-truth strings

References
----------
Luo, Y. et al. (2024). Graph-constrained Reasoning: Faithful Reasoning on
Knowledge Graphs with Large Language Models.
"""

from __future__ import annotations

import re
from typing import List, Set, Dict, Optional

import networkx as nx


# ===========================================================================
# 1.  CORRECTNESS METRICS
# ===========================================================================

def compute_hit(
    generated_paths: List[str],
    ground_truth_paths: List[str],
) -> float:
    """
    Hit@K: 1.0 if *any* generated path exactly matches *any* ground-truth
    path, 0.0 otherwise.

    This directly mirrors the Hit metric used by Luo et al. (2024) on
    WebQSP/CWQ, adapted to path-level matching rather than answer-entity
    matching.

    Parameters
    ----------
    generated_paths   : Decoded path strings produced by the model.
    ground_truth_paths: Linearised ground-truth paths from the eval dataset.
    """
    gt_set = set(p.strip() for p in ground_truth_paths)
    for gen in generated_paths:
        if gen.strip() in gt_set:
            return 1.0
    return 0.0


def _token_f1(pred: str, gold: str) -> float:
    """Token-overlap F1 between two strings (case-insensitive)."""
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    common = pred_set & gold_set
    if not common:
        return 0.0
    precision = len(common) / len(pred_set)
    recall = len(common) / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def compute_path_f1(
    generated_paths: List[str],
    ground_truth_paths: List[str],
) -> float:
    """
    Path-level F1: maximum token-overlap F1 between the best generated path
    and any ground-truth path.

    Analogous to the F1 metric in Luo et al. (2024), which balances precision
    and recall over the set of answer entities.  Here we operate over the
    token sequences of linearised paths instead.
    """
    if not generated_paths or not ground_truth_paths:
        return 0.0
    best = 0.0
    for gen in generated_paths:
        for gt in ground_truth_paths:
            best = max(best, _token_f1(gen, gt))
    return best


def compute_activity_sequence_accuracy(
    generated_paths: List[str],
    ground_truth_paths: List[str],
) -> float:
    """
    Check whether the ordered sequence of *activity labels* in any generated
    path matches that of any ground-truth path, ignoring object-type tokens
    and edge labels.

    This is more lenient than exact Hit because it abstracts away qualifiers,
    making it suitable for comparing paths that differ only in object types
    (common in OCEL 2.0 where the same activity touches many object types).
    """
    def extract_activities(path: str) -> List[str]:
        # Tokens of the form  Event:<ActivityName>
        return re.findall(r"Event:(\S+)", path)

    gt_act_seqs = [extract_activities(p) for p in ground_truth_paths]
    for gen in generated_paths:
        gen_acts = extract_activities(gen)
        if gen_acts in gt_act_seqs:
            return 1.0
    return 0.0


# ===========================================================================
# 2.  FAITHFULNESS METRICS
# ===========================================================================

def _parse_path_nodes_and_edges(path_str: str):
    """
    Parse a linearised path string back into (node_labels, edge_labels).

    Supports two serialisation formats used in the codebase:
      - linearize_path:     "Event:A rel1 Object:B rel2 Event:C"
      - serialize_ocel_path_v2: "Event:A --(rel1)--> Object:B --(rel2)--> Event:C"
    """
    # Normalise v2 arrow syntax to plain tokens
    normalised = re.sub(r"\s*--\(([^)]+)\)-->\s*", r" \1 ", path_str)
    tokens = normalised.strip().split()

    nodes = []
    edges = []
    for tok in tokens:
        if tok.startswith("Event:") or tok.startswith("Object:"):
            nodes.append(tok)
        else:
            edges.append(tok)
    return nodes, edges


def _build_valid_semantic_edges(G: nx.DiGraph) -> Set[tuple]:
    """
    Pre-compute the set of valid (src_label, rel, tgt_label) triples from G.
    Uses the same node_semantic_label scheme as the trie construction.
    """
    from utils.generate_eval_dataset import node_semantic_label

    valid = set()
    for u, v, data in G.edges(data=True):
        rel = data.get("label", "rel").replace(" ", "_")
        valid.add((node_semantic_label(G, u), rel, node_semantic_label(G, v)))
    return valid


def compute_constraint_compliance(
    generated_paths: List[str],
    G: nx.DiGraph,
    valid_edges: Optional[Set[tuple]] = None,
) -> float:
    """
    Fraction of generated paths where *every* consecutive (node, edge, node)
    triple is present in the OCEL graph.

    For GCR (constrained), this should be 1.0 by construction.
    For the unconstrained baseline, this measures hallucination.

    Corresponds to the "faithful reasoning ratio" in Luo et al. (2024, §5.3).

    Parameters
    ----------
    generated_paths : Decoded path strings.
    G               : The OCEL graph used for trie construction.
    valid_edges     : Pre-computed edge set (pass once; computed lazily otherwise).
    """
    if valid_edges is None:
        valid_edges = _build_valid_semantic_edges(G)

    if not generated_paths:
        return 0.0

    compliant = 0
    for path_str in generated_paths:
        nodes, edges = _parse_path_nodes_and_edges(path_str)
        # A path needs at least one edge to be verifiable
        if len(nodes) < 2:
            compliant += 1  # trivial single-node path — count as compliant
            continue
        path_ok = True
        for i in range(len(edges)):
            if i + 1 >= len(nodes):
                break
            triple = (nodes[i], edges[i], nodes[i + 1])
            if triple not in valid_edges:
                path_ok = False
                break
        if path_ok:
            compliant += 1

    return compliant / len(generated_paths)


def compute_hallucination_rate(
    generated_paths: List[str],
    G: nx.DiGraph,
    valid_edges: Optional[Set[tuple]] = None,
) -> float:
    """Complement of constraint compliance: fraction of paths with ≥1 hallucinated edge."""
    return 1.0 - compute_constraint_compliance(generated_paths, G, valid_edges)


# ===========================================================================
# 3.  PROCESS-SPECIFIC METRICS
# ===========================================================================

def compute_temporal_validity(
    generated_paths: List[str],
    G: nx.DiGraph,
) -> float:
    """
    For each generated path, check whether every consecutive pair of event
    nodes appears in non-decreasing timestamp order in the graph.

    A path that traverses events out of causal order is semantically wrong
    even if structurally valid (all edges exist in the graph).

    Returns the fraction of generated paths that are temporally valid.
    """
    from utils.generate_eval_dataset import node_semantic_label

    # Build a lookup: semantic_label -> list of (node_id, timestamp)
    label_to_ts: Dict[str, List] = {}
    for n, data in G.nodes(data=True):
        if data.get("entity_type") == "Event":
            label = node_semantic_label(G, n)
            ts_str = data.get("timestamp", "")
            label_to_ts.setdefault(label, []).append(ts_str)

    if not generated_paths:
        return 0.0

    valid_count = 0
    for path_str in generated_paths:
        event_labels = re.findall(r"Event:\S+", path_str)
        if len(event_labels) < 2:
            valid_count += 1  # nothing to compare
            continue

        path_valid = True
        for i in range(len(event_labels) - 1):
            ts_a_list = label_to_ts.get(event_labels[i], [])
            ts_b_list = label_to_ts.get(event_labels[i + 1], [])
            # If timestamps are available, check at least one valid ordering exists
            if ts_a_list and ts_b_list:
                # Accept if any combination is non-decreasing
                ok = any(
                    str(a) <= str(b)
                    for a in ts_a_list
                    for b in ts_b_list
                )
                if not ok:
                    path_valid = False
                    break
        if path_valid:
            valid_count += 1

    return valid_count / len(generated_paths)


def compute_lifecycle_coverage(
    generated_paths: List[str],
    ground_truth_events: List[str],
    G: nx.DiGraph,
) -> float:
    """
    Fraction of the object's ground-truth lifecycle events (activity labels)
    that appear in *any* of the generated paths.

    Measures whether the generated paths cover the relevant portion of the
    object's end-to-end process, a notion specific to object-centric event
    logs (OCEL 2.0) with no direct equivalent in KG-QA evaluation.

    Parameters
    ----------
    generated_paths       : Decoded path strings.
    ground_truth_events   : List of event node IDs from expected_outputs.answer.events.
    G                     : The OCEL graph.
    """
    if not ground_truth_events:
        return 1.0  # vacuously true

    # Convert ground-truth event node IDs to activity labels
    gt_activities: Set[str] = set()
    for ev_id in ground_truth_events:
        act = G.nodes[ev_id].get("activity", "") if ev_id in G.nodes else ""
        if act:
            gt_activities.add(f"Event:{act.replace(' ', '_')}")

    if not gt_activities:
        return 0.0

    # Collect all activity labels mentioned across generated paths
    generated_activities: Set[str] = set()
    for path_str in generated_paths:
        generated_activities.update(re.findall(r"Event:\S+", path_str))

    covered = gt_activities & generated_activities
    return len(covered) / len(gt_activities)


# ===========================================================================
# 4.  AGGREGATE SCORER
# ===========================================================================

def score_single(
    generated_paths: List[str],
    item: dict,
    G: nx.DiGraph,
    valid_edges: Optional[Set[tuple]] = None,
) -> dict:
    """
    Compute all metrics for a single eval item and return them as a dict.

    Parameters
    ----------
    generated_paths : Output of the model under evaluation.
    item            : One record from eval_local.jsonl or eval_localobj.jsonl.
    G               : The OCEL graph.
    valid_edges     : Pre-computed edge set (pass for efficiency in loops).
    """
    gt_paths = item["expected_outputs"]["paths"]
    gt_events = item["expected_outputs"]["answer"].get("events", [])

    return {
        "id": item["id"],
        "task_type": item["task_type"],
        # --- Correctness ---
        "hit": compute_hit(generated_paths, gt_paths),
        "path_f1": compute_path_f1(generated_paths, gt_paths),
        "activity_accuracy": compute_activity_sequence_accuracy(
            generated_paths, gt_paths
        ),
        # --- Faithfulness ---
        "constraint_compliance": compute_constraint_compliance(
            generated_paths, G, valid_edges
        ),
        "hallucination_rate": compute_hallucination_rate(
            generated_paths, G, valid_edges
        ),
        # --- Process-specific ---
        "temporal_validity": compute_temporal_validity(generated_paths, G),
        "lifecycle_coverage": compute_lifecycle_coverage(
            generated_paths, gt_events, G
        ),
    }


def aggregate_scores(scores: List[dict]) -> dict:
    """Macro-average all numeric metrics across the score list."""
    if not scores:
        return {}
    numeric_keys = [
        k for k, v in scores[0].items() if isinstance(v, (int, float))
    ]
    return {
        k: sum(s[k] for s in scores) / len(scores)
        for k in numeric_keys
    }
