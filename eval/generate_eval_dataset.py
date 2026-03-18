"""
utils/generate_eval_dataset.py
-------------------------------
Generates the evaluation dataset from an OCEL 2.0 NetworkX graph.

Serialization contract
-----------------------
Ground-truth paths are serialized using the *same* canonical format as the
trie construction in gcr/gcr.py:

    label(n0) rel_01 label(n1) rel_12 label(n2) ...

where label() = "Event:<Activity_With_Underscores>" or
               "Object:<object_type_with_underscores>".

This ensures that Hit and F1 comparisons between generated and ground-truth
paths are meaningful (no space/underscore mismatch, no duplicate interior
nodes).

Fixes vs. original
------------------
1. node_semantic_label / linearize_triplets now delegate to the shared
   node_label() from gcr.gcr, eliminating any risk of future drift.

2. linearize_triplets now produces chain-style output (no duplicate interior
   nodes), consistent with the fixed linearize_path in gcr/gcr.py.

3. Added fixed random seed, stratified sampling, and increased default sample
   sizes for reproducibility and coverage.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import List, Dict

import networkx as nx

# Import the single source-of-truth label function from gcr.gcr
from gcr.gcr import node_label, extract_paths


# ---------------------------------------------------------------------------
# Canonical label (thin wrapper kept for backward compatibility)
# ---------------------------------------------------------------------------

def node_semantic_label(G: nx.DiGraph, n: str) -> str:
    """Alias for gcr.gcr.node_label — do not duplicate the logic here."""
    return node_label(G, n)


# ---------------------------------------------------------------------------
# Ground-truth path linearization  (chain format, matching trie format)
# ---------------------------------------------------------------------------

def linearize_triplets(G: nx.DiGraph, trip_path: List[tuple]) -> str:
    """
    Convert a list of (u, rel, v) triples to a canonical chain string:

        label(u0) rel_01 label(u1) rel_12 label(u2) ...

    Interior nodes are emitted once only.  This matches the format produced
    by gcr.gcr.linearize_path, ensuring Hit/F1 comparisons are valid.
    """
    if not trip_path:
        return ""
    parts: List[str] = [node_label(G, trip_path[0][0])]
    for u, rel, v in trip_path:
        parts.append(rel)
        parts.append(node_label(G, v))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Stratified sampling helpers
# ---------------------------------------------------------------------------

def _stratified_event_sample(
    G: nx.DiGraph,
    num: int,
    rng: random.Random,
) -> List[str]:
    """
    Sample *num* event nodes so that every activity type appears at least once.
    Falls back to uniform random sampling once all activity types are covered.
    """
    events = [n for n in G.nodes if G.nodes[n].get("entity_type") == "Event"]
    if not events:
        return []

    by_activity: Dict[str, List[str]] = defaultdict(list)
    for ev in events:
        act = G.nodes[ev].get("activity", "Unknown")
        by_activity[act].append(ev)

    selected: List[str] = []
    seen: set = set()

    for act_nodes in by_activity.values():
        candidate = rng.choice(act_nodes)
        if candidate not in seen:
            selected.append(candidate)
            seen.add(candidate)
        if len(selected) >= num:
            break

    remaining = [e for e in events if e not in seen]
    rng.shuffle(remaining)
    for e in remaining:
        if len(selected) >= num:
            break
        selected.append(e)
        seen.add(e)

    return selected[:num]


def _stratified_object_sample(
    G: nx.DiGraph,
    num: int,
    rng: random.Random,
) -> List[str]:
    """
    Sample *num* object nodes so that every object type appears at least once.
    """
    objects = [n for n in G.nodes if G.nodes[n].get("entity_type") == "Object"]
    if not objects:
        return []

    by_type: Dict[str, List[str]] = defaultdict(list)
    for obj in objects:
        otype = G.nodes[obj].get("object_type", "Unknown")
        by_type[otype].append(obj)

    selected: List[str] = []
    seen: set = set()

    for type_nodes in by_type.values():
        candidate = rng.choice(type_nodes)
        if candidate not in seen:
            selected.append(candidate)
            seen.add(candidate)
        if len(selected) >= num:
            break

    remaining = [o for o in objects if o not in seen]
    rng.shuffle(remaining)
    for o in remaining:
        if len(selected) >= num:
            break
        selected.append(o)
        seen.add(o)

    return selected[:num]


# ---------------------------------------------------------------------------
# Ground-truth path extraction
# ---------------------------------------------------------------------------

def extract_semantic_paths(
    G: nx.DiGraph,
    start_node: str,
    max_depth: int = 4,
) -> List[List[tuple]]:
    """
    Wrapper around gcr.gcr.extract_paths returning (u, rel, v) triple lists.
    """
    return extract_paths(G, start_node, max_depth=max_depth)


# ---------------------------------------------------------------------------
# Question builders
# ---------------------------------------------------------------------------

def make_local_questions(
    G: nx.DiGraph,
    num: int = 100,
    seed: int = 42,
    max_gt_paths: int = 10,
) -> List[Dict]:
    """
    Build *num* local event queries.

    Ground-truth paths use linearize_triplets (chain format, underscores),
    which is identical to what GCRProcessAgent will generate via the trie.
    """
    rng = random.Random(seed)
    sampled_events = _stratified_event_sample(G, num, rng)
    qs = []

    for i, ev in enumerate(sampled_events):
        activity = G.nodes[ev].get("activity", "Unknown")
        q = f"What happens after event {ev}? What objects are involved?"

        gt_paths = [
            linearize_triplets(G, p)
            for p in extract_semantic_paths(G, ev, max_depth=3)
        ]

        linked_objs = [
            obj
            for _, obj, _ in G.out_edges(ev, data=True)
            if G.nodes[obj].get("entity_type") == "Object"
        ]

        qs.append({
            "id": f"LOCAL_{i:03d}",
            "task_type": "local_event_query",
            "question": q,
            "topic_entities": [ev],
            "expected_outputs": {
                "paths": gt_paths[:max_gt_paths],
                "answer": {
                    "activity": activity,
                    "objects": linked_objs,
                },
            },
        })

    return qs


def make_local_object_questions(
    G: nx.DiGraph,
    num: int = 100,
    seed: int = 42,
    max_gt_paths: int = 10,
) -> List[Dict]:
    """
    Build *num* local object lifecycle queries.
    """
    rng = random.Random(seed)
    sampled_objects = _stratified_object_sample(G, num, rng)
    qs = []

    for i, obj in enumerate(sampled_objects):
        otype = G.nodes[obj].get("object_type", "Unknown")
        q = f"What are the events associated with object {obj}? Describe its lifecycle."

        events = list({
            u
            for u, _, _ in G.in_edges(obj, data=True)
            if G.nodes[u].get("entity_type") == "Event"
        } | {
            v
            for _, v, _ in G.out_edges(obj, data=True)
            if G.nodes[v].get("entity_type") == "Event"
        })

        sem_paths = [
            linearize_triplets(G, p)
            for p in extract_semantic_paths(G, obj, max_depth=4)
        ]

        qs.append({
            "id": f"OBJ_{i:03d}",
            "task_type": "local_object_query",
            "question": q,
            "topic_entities": [obj],
            "expected_outputs": {
                "paths": sem_paths[:max_gt_paths],
                "answer": {
                    "object_type": otype,
                    "events": events,
                    "num_events": len(events),
                },
            },
        })

    return qs


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_all_datasets(
    G_ocel: nx.DiGraph,
    out_prefix: str = "eval",
    num_local: int = 100,
    num_object: int = 100,
    seed: int = 42,
) -> None:
    """
    Write eval_local.jsonl and eval_localobj.jsonl to disk.

    Parameters
    ----------
    seed : Random seed — document this value in the thesis methods section
           for full reproducibility.
    """
    local = make_local_questions(G_ocel, num=num_local, seed=seed)
    localobj = make_local_object_questions(G_ocel, num=num_object, seed=seed)

    local_path = f"{out_prefix}_local.jsonl"
    obj_path = f"{out_prefix}_localobj.jsonl"

    with open(local_path, "w") as f:
        for q in local:
            f.write(json.dumps(q) + "\n")

    with open(obj_path, "w") as f:
        for q in localobj:
            f.write(json.dumps(q) + "\n")

    print(f"Datasets generated (seed={seed}):")
    print(f"  {local_path}  ({len(local)} questions)")
    print(f"  {obj_path}  ({len(localobj)} questions)")
