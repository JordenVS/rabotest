import json
import random
import networkx as nx
from typing import List, Dict

# ========================================
# Utility: semantic labeling (same as trie)
# ========================================

def node_semantic_label(G, n):
    t = G.nodes[n].get("entity_type")
    if t == "Event":
        act = G.nodes[n].get("activity", "Unknown").replace(" ", "_")
        return f"Event:{act}"
    elif t == "Object":
        otype = G.nodes[n].get("object_type", "Object").replace(" ", "_")
        return f"Object:{otype}"
    else:
        return n


# ========================================
# Path extraction for ground truth
# ========================================

def extract_semantic_paths(G, start_node, max_depth=4):
    paths = []
    stack = [(start_node, [], 0, {start_node})]
    while stack:
        node, cur, d, visited = stack.pop()
        if d >= max_depth or G.out_degree(node) == 0:
            if cur:
                paths.append(cur)
            continue
        for _, nxt, data in G.out_edges(node, data=True):
            rel = data.get("label", "rel").replace(" ", "_")
            if nxt in visited:
                paths.append(cur)
                continue
            stack.append((nxt, cur + [(node, rel, nxt)], d + 1, visited | {nxt}))
    return paths


def linearize_triplets(G, trip_path):
    parts = []
    for u, rel, v in trip_path:
        parts.append(node_semantic_label(G, u))
        parts.append(rel)
        parts.append(node_semantic_label(G, v))
    return " ".join(parts)


# ========================================
# Local questions
# ========================================

def make_local_questions(G, num=20):
    events = [n for n in G.nodes if G.nodes[n].get("entity_type") == "Event"]
    qs = []

    for i in range(num):
        ev = random.choice(events)
        activity = G.nodes[ev].get("activity")

        q = f"What happens after event {ev}? What objects are involved?"
        topic = [ev]

        # ground truth semantic paths
        paths = [linearize_triplets(G, p) for p in extract_semantic_paths(G, ev, max_depth=3)]

        # objects linked to event
        objs = []
        for _, obj, d in G.out_edges(ev, data=True):
            if G.nodes[obj].get("entity_type") == "Object":
                objs.append(obj)

        answer = {
            "activity": activity,
            "objects": objs
        }

        qs.append({
            "id": f"LOCAL_{i:03d}",
            "task_type": "local_event_query",
            "question": q,
            "topic_entities": topic,
            "expected_outputs": {
                "paths": paths[:5],
                "answer": answer
            }
        })
    return qs

def make_local_object_questions(G, num=20):
    objects = [n for n in G.nodes if G.nodes[n].get("entity_type") == "Object"]
    qs = []

    for i in range(num):
        obj = random.choice(objects)
        otype = G.nodes[obj].get("object_type")

        q = f"What are the events associated with object {obj}? Describe its lifecycle."
        topic = [obj]

        # Extract all event relations for ground truth
        events = []
        for u, v, data in G.in_edges(obj, data=True):
            if G.nodes[u].get("entity_type") == "Event":
                events.append(u)
        for u, v, data in G.out_edges(obj, data=True):
            if G.nodes[v].get("entity_type") == "Event":
                events.append(v)

        # Semantic paths starting from the object
        sem_paths = [
            linearize_triplets(G, p)
            for p in extract_semantic_paths(G, obj, max_depth=4)
        ]

        answer = {
            "object_type": otype,
            "events": events,
            "num_events": len(events)
        }

        qs.append({
            "id": f"OBJ_{i:03d}",
            "task_type": "local_object_query",
            "question": q,
            "topic_entities": topic,
            "expected_outputs": {
                "paths": sem_paths[:5],
                "answer": answer
            }
        })

    return qs

# ========================================
# Global questions (Petri net)
# ========================================

def make_global_questions(petri_model, num=20):
    transitions = list(petri_model.transitions)
    qs = []

    for i in range(num):
        t = random.choice(transitions)
        label = t.label.replace(" ", "_")

        q = f"What are plausible next steps after transition {label} in the global model?"
        topic = [label]

        # ground truth: check outgoing arcs
        next_trans = []
        for arc in petri_model.arcs:
            if arc.source == t and hasattr(arc.target, "out_arcs"):
                for arc2 in arc.target.out_arcs:
                    if hasattr(arc2.target, "label"):
                        next_trans.append(arc2.target.label.replace(" ", "_"))

        answer = {"next_transitions": next_trans}

        qs.append({
            "id": f"GLOBAL_{i:03d}",
            "task_type": "global_transition_query",
            "question": q,
            "topic_entities": topic,
            "expected_outputs": {
                "paths": [],      # paths are optional for global model
                "answer": answer
            }
        })
    return qs


# ========================================
# Cross-level questions
# ========================================

def make_cross_level_questions(G, petri_model, num=20):
    events = [n for n in G.nodes if G.nodes[n].get("entity_type") == "Event"]
    qs = []

    for i in range(num):
        ev = random.choice(events)
        act = G.nodes[ev].get("activity", "UnknownActivity")

        q = f"Does event {ev} conform to the global model? Where does it fit?"
        topic = [ev]

        # Map to global transition label
        global_label = act.replace(" ", "_")

        # ground truth: outgoing
        semantic_paths = [linearize_triplets(G, p) for p in extract_semantic_paths(G, ev)]

        qs.append({
            "id": f"CROSS_{i:03d}",
            "task_type": "local_global_alignment",
            "question": q,
            "topic_entities": topic,
            "expected_outputs": {
                "paths": semantic_paths[:5],
                "answer": {
                    "mapped_transition": global_label
                }
            }
        })
    return qs


# ========================================
# Main: build dataset files
# ========================================

def build_all_datasets(G_ocel, petri_model, out_prefix="eval"):
    local = make_local_questions(G_ocel)
    global_q = make_global_questions(petri_model)
    cross = make_cross_level_questions(G_ocel, petri_model)

    with open(f"{out_prefix}_local.jsonl", "w") as f:
        for q in local:
            f.write(json.dumps(q) + "\n")

    # with open(f"{out_prefix}_global.jsonl", "w") as f:
    #     for q in global_q:
    #         f.write(json.dumps(q) + "\n")

    # with open(f"{out_prefix}_cross.jsonl", "w") as f:
    #     for q in cross:
    #         f.write(json.dumps(q) + "\n")

    print("Datasets generated:")
    print(f" - {out_prefix}_local.jsonl")
    # print(f" - {out_prefix}_global.jsonl")
    # print(f" - {out_prefix}_cross.jsonl")