"""
eval/generate_predicted_paths.py
=================================
Stage 1 of the two-stage GCR evaluation pipeline.

Runs the path-generation LLM under graph-constrained decoding (Luo et al., 2025)
over the evaluation sample and writes two JSONL files:

    <out_dir>/predicted_paths_constrained.jsonl
    <out_dir>/predicted_paths_unconstrained.jsonl   (unless --skip_unconstrained)

Architecture
------------
Two graphs serve distinct roles:

  G_context  — heterogeneous graph (Event + Object nodes, participation edges).
               ``build_events_dict_from_context_graph`` reads this to reconstruct
               which objects each event involves, without reloading the raw OCEL file.

  G_behavior — event-only graph (Event nodes, behavior edges encoding process sequences).
               ``build_event_successors_from_g_behavior`` reads this to derive which
               events can legally follow a given event, forming the basis of the
               ProcessTrie used to constrain decoding.

Together they allow ``GCRProcessAgent`` to enumerate only behaviorally valid paths
that remain anchored to the objects of the query entity, producing zero-hallucination
reasoning paths as described in Luo et al. (2025).

Output schema (one JSON object per line)
-----------------------------------------
{
  "id":            "Q007",
  "paths":         ["Event:Create_Purchase_Order Event:Approve_Purchase_Order ..."],
  "trie_build_s":  0.12,
  "generation_s":  1.45,
  "total_s":       1.57,
  "note":          ""      # non-empty only on error / skip
}

Usage
-----
    python -m eval.generate_predicted_paths \\
        --dataset         eval/sampled_100.json \\
        --graph_context   context_graph.graphml \\
        --graph_behavior  behavior_graph.graphml \\
        --model           Qwen/Qwen2.5-1.5B-Instruct \\
        --out_dir         results \\
        --num_paths       3 \\
        --max_depth       4

        python -m eval.generate_predicted_paths --dataset eval/data/sampled_100.json --graph_context graphs/context_graph.graphml --graph_behavior graphs/behavior_graph.graphml --model Qwen/Qwen2.5-1.5B-Instruct --out_dir paths_new_d3 --num_paths 3 --max_depth 3 --limit 5
    Quick test (5 questions only):
        --limit 5

References
----------
Luo, L., Zhao, Z., Haffari, G., Li, Y.-F., Gong, C., & Pan, S. (2025).
    Graph-constrained reasoning: Faithful reasoning on knowledge graphs with
    large language models. ICML 2025.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root resolution (works regardless of working directory)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.graph_utils import load_graphml_to_networkx
from gcr.gcr import build_events_dict_from_context_graph, build_event_successors_from_g_behavior
from gcr.processors import GCRProcessAgent


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> List[Dict]:
    """Load the evaluation sample — accepts both a JSON array and JSONL."""
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def _load_done_ids(path: str) -> set:
    """Return the set of instance_ids already written to *path* (for resume support)."""
    done: set = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["instance_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _skip_record(instance_id: str, reason: str) -> Dict:
    return {
        "instance_id":   instance_id,
        "paths":         [],
        "trie_build_s":  0.0,
        "generation_s":  0.0,
        "total_s":       0.0,
        "prompt_tokens": 0,
        "note":          reason,
    }


# ---------------------------------------------------------------------------
# Core generation loop
# ---------------------------------------------------------------------------

def generate_paths(
    questions: List[Dict],
    agent: GCRProcessAgent,
    *,
    constrained: bool,
    num_paths: int,
    max_depth: int,
    out_path: str,
) -> None:
    """
    Run *agent* over *questions* and write one record per question to *out_path*.

    The file is opened in append mode so an interrupted run can be resumed:
    questions whose instance_id is already present in *out_path* are skipped.

    The anchor object is read from ``anchor_object.oid``, matching the
    dataset schema produced by build_evaluation_dataset().

    Parameters
    ----------
    constrained:
        True  → graph-constrained beam search (GCR proper).
        False → unconstrained beam search (ablation baseline;
                 'GCR w/o constraint' in Luo et al., 2025, Fig. 5).
    """
    label = "constrained" if constrained else "unconstrained"
    done = _load_done_ids(out_path)

    skipped = sum(1 for q in questions if q.get("instance_id") in done)
    if skipped:
        print(f"  [{label}] Resuming — skipping {skipped} already-completed questions.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "a", encoding="utf-8") as f_out:
        for q in tqdm(questions, desc=f"[{label}]"):
            instance_id = q.get("instance_id", "???")

            if instance_id in done:
                continue

            anchor = q.get("anchor_object", {}).get("oid")

            if not anchor:
                rec = _skip_record(instance_id, "anchor_object.oid missing")
            else:
                try:
                    timing = agent.timed_generate(
                        anchor_object=anchor,
                        question=q["question"],
                        constrained=constrained,
                        num_paths=num_paths,
                        max_depth=max_depth,
                    )
                    rec = {
                        "instance_id": instance_id,
                        "paths":       timing.pop("paths"),
                        "note":        "",
                        **timing,
                    }
                except Exception as exc:
                    print(f"\n  [WARNING] {instance_id} ({anchor}): {exc}")
                    rec = _skip_record(instance_id, f"error: {exc}")

            f_out.write(json.dumps(rec, default=str) + "\n")
            f_out.flush()

    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage 1: generate GCR reasoning-path predictions "
            "(constrained + unconstrained ablation) over the evaluation sample."
        )
    )
    p.add_argument(
        "--dataset", required=True,
        help="Evaluation sample — JSON array or JSONL (e.g. eval/sampled_100.json)"
    )
    p.add_argument(
        "--graph_context", required=True,
        help=(
            "Context graph GraphML (Event + Object nodes, participation edges). "
            "Used to reconstruct event-object memberships."
        )
    )
    p.add_argument(
        "--graph_behavior", required=True,
        help=(
            "Behavior graph GraphML (Event nodes only, behavior edges). "
            "Used to derive valid event successors for trie construction."
        )
    )
    p.add_argument(
        "--model", default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID for the path-generation LLM"
    )
    p.add_argument(
        "--device", default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Torch device"
    )
    p.add_argument(
        "--num_paths", type=int, default=3,
        help="Number of beam paths per question (num_beams in beam search)"
    )
    p.add_argument(
        "--max_depth", type=int, default=4,
        help="Maximum path depth / trie hops"
    )
    p.add_argument(
        "--out_dir", default="results",
        help="Output directory for the predicted-path JSONL files"
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Use only the first N questions — useful for quick testing"
    )
    p.add_argument(
        "--skip_unconstrained", action="store_true",
        help="Skip the unconstrained ablation run (saves time)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load graphs
    # ------------------------------------------------------------------ #
    print(f"Loading context graph:  {args.graph_context}")
    G_context = load_graphml_to_networkx(args.graph_context)

    print(f"Loading behavior graph: {args.graph_behavior}")
    G_behavior = load_graphml_to_networkx(args.graph_behavior)

    # ------------------------------------------------------------------ #
    # 2. Load evaluation sample
    # ------------------------------------------------------------------ #
    questions = load_dataset(args.dataset)
    if args.limit:
        questions = questions[: args.limit]
        print(f"[--limit] Using first {args.limit} questions.")
    print(f"Loaded {len(questions)} questions from {args.dataset}.\n")

    # ------------------------------------------------------------------ #
    # 3. Build GCR agent
    #    The model is loaded once and reused for both the constrained and
    #    unconstrained runs to avoid double-loading weights.
    # ------------------------------------------------------------------ #
    print("Building events dict from context graph…")
    events = build_events_dict_from_context_graph(G_context)
    print(f"  {len(events)} events loaded.")

    print("Building event successors from behavior graph…")
    event_successors = build_event_successors_from_g_behavior(G_behavior, events)
    print(f"  {len(event_successors)} events have successors.")

    print(f"Loading path-generation model: {args.model}")
    agent = GCRProcessAgent(
        model_id=args.model,
        events=events,
        event_successors=event_successors,
        device=args.device,
    )
    print("Agent ready.\n")

    # ------------------------------------------------------------------ #
    # 4. Constrained run (GCR proper)
    # ------------------------------------------------------------------ #
    print("=== Constrained (GCR) ===")
    generate_paths(
        questions, agent,
        constrained=True,
        num_paths=args.num_paths,
        max_depth=args.max_depth,
        out_path=os.path.join(args.out_dir, "predicted_paths_constrained.jsonl"),
    )

    # ------------------------------------------------------------------ #
    # 5. Unconstrained ablation
    # ------------------------------------------------------------------ #
    if not args.skip_unconstrained:
        print("\n=== Unconstrained (ablation baseline) ===")
        generate_paths(
            questions, agent,
            constrained=False,
            num_paths=args.num_paths,
            max_depth=args.max_depth,
            out_path=os.path.join(args.out_dir, "predicted_paths_unconstrained.jsonl"),
        )

    # ------------------------------------------------------------------ #
    # 6. Summary
    # ------------------------------------------------------------------ #
    print("\n=== Done. Pass these files to run_evaluation.py: ===")
    print(f"  --constrained_paths   {args.out_dir}/predicted_paths_constrained.jsonl")
    if not args.skip_unconstrained:
        print(f"  --unconstrained_paths {args.out_dir}/predicted_paths_unconstrained.jsonl")


if __name__ == "__main__":
    main()