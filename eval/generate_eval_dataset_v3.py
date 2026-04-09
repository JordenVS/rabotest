import random
import uuid
from collections import defaultdict

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