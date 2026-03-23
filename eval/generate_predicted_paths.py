"""
eval/generate_predicted_paths.py
==================================
Runs the small path-generation LLM over the evaluation dataset to produce
the predicted-paths JSONL files required by run_evaluation.py.

Two files are written:
    <out_dir>/predicted_paths_constrained.jsonl
    <out_dir>/predicted_paths_unconstrained.jsonl

Each file has one record per question:
    {"id": "T1_001", "paths": ["Event:Create_Purchase_Order NEXT_FOR_po ..."]}

These files are then passed to run_evaluation.py via:
    --constrained_paths   <out_dir>/predicted_paths_constrained.jsonl
    --unconstrained_paths <out_dir>/predicted_paths_unconstrained.jsonl

The path-generation LLM (small, e.g. Qwen2.5-1.5B-Instruct) is intentionally
separate from the answer-generation LLM (large, e.g. Qwen2.5-7B-Instruct) so
that the two-stage pipeline mirrors the intended GCR architecture: constrained
decoding over a small model produces structured process paths, which are then
used as grounded context for a larger model to generate natural language answers.

Usage
-----
    python -m eval.generate_predicted_paths \\
        --dataset  eval/data/eval_combined.jsonl \\
        --graphml  test2.graphml \\
        --model    Qwen/Qwen2.5-1.5B-Instruct \\
        --out_dir  results \\
        --num_paths 3 \\
        --max_depth 4

    For quick testing:
        --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve project root so imports work regardless of cwd
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.graph_utils import load_graphml_to_networkx
from gcr.processors import GCRProcessAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved {len(records)} records → {path}")


# ---------------------------------------------------------------------------
# Path generation
# ---------------------------------------------------------------------------

def generate_paths(
    questions: List[Dict],
    agent: GCRProcessAgent,
    constrained: bool,
    num_paths: int,
    max_depth: int,
) -> List[Dict]:
    """
    Run the agent over all questions and return a list of
    {"id": ..., "paths": [...], "trie_build_s": ..., "generation_s": ...} records.

    The seed entity for path generation is always topic_entities[0], matching
    the convention used in run_evaluation.py.  Questions with no topic entity
    are skipped and recorded with an empty path list.
    """
    mode = "constrained" if constrained else "unconstrained"
    records: List[Dict] = []

    for q in tqdm(questions, desc=f"Generating {mode} paths"):
        topic_entities = q.get("topic_entities", [])
        seed_entity = topic_entities[0] if topic_entities else None

        if seed_entity is None or seed_entity not in agent.graph:
            records.append({
                "id": q["id"],
                "paths": [],
                "trie_build_s": 0.0,
                "generation_s": 0.0,
                "total_s": 0.0,
                "prompt_tokens": 0,
                "note": "seed entity missing or not in graph",
            })
            continue

        try:
            timing = agent.timed_generate(
                seed_entity=seed_entity,
                question=q["question"],
                constrained=constrained,
                num_paths=num_paths,
                max_depth=max_depth,
            )
            records.append({
                "id": q["id"],
                "paths": timing.pop("paths"),
                **timing,
            })
        except Exception as e:
            print(f"\n  [WARNING] {q['id']} ({seed_entity}) failed: {e}")
            records.append({
                "id": q["id"],
                "paths": [],
                "trie_build_s": 0.0,
                "generation_s": 0.0,
                "total_s": 0.0,
                "prompt_tokens": 0,
                "note": f"error: {e}",
            })

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate constrained and unconstrained predicted-path files for GCR evaluation."
    )
    p.add_argument(
        "--dataset", required=True,
        help="Path to the combined evaluation JSONL (e.g. eval/data/eval_combined.jsonl)"
    )
    p.add_argument(
        "--graphml", required=True,
        help="Path to the OCEL process graph (e.g. test2.graphml)"
    )
    p.add_argument(
        "--model", default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID for the small path-generation LLM"
    )
    p.add_argument(
        "--device", default="cpu",
        help="Device for the path-generation model ('cpu', 'cuda', 'mps')"
    )
    p.add_argument(
        "--num_paths", type=int, default=3,
        help="Number of beam paths to generate per question (num_beams)"
    )
    p.add_argument(
        "--max_depth", type=int, default=4,
        help="Maximum trie depth / path hops"
    )
    p.add_argument(
        "--out_dir", default="results",
        help="Directory where the two predicted-paths JSONL files are written"
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of questions (useful for quick testing)"
    )
    p.add_argument(
        "--skip_unconstrained", action="store_true",
        help="Only generate the constrained file (saves time if ablation not needed)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load graph
    print(f"Loading graph: {args.graphml}")
    G = load_graphml_to_networkx(args.graphml)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load dataset
    questions = load_jsonl(args.dataset)
    if args.limit:
        questions = questions[:args.limit]
    print(f"Loaded {len(questions)} questions from {args.dataset}.")

    # Load agent once — shared for both runs
    print(f"Loading path-generation model: {args.model}")
    agent = GCRProcessAgent(args.model, G, device=args.device)
    print("  Agent ready.\n")

    # --- Constrained ---
    print("--- Generating CONSTRAINED paths ---")
    constrained_records = generate_paths(
        questions, agent,
        constrained=True,
        num_paths=args.num_paths,
        max_depth=args.max_depth,
    )
    save_jsonl(
        constrained_records,
        os.path.join(args.out_dir, "predicted_paths_constrained.jsonl"),
    )

    # --- Unconstrained ---
    if not args.skip_unconstrained:
        print("\n--- Generating UNCONSTRAINED paths ---")
        unconstrained_records = generate_paths(
            questions, agent,
            constrained=False,
            num_paths=args.num_paths,
            max_depth=args.max_depth,
        )
        save_jsonl(
            unconstrained_records,
            os.path.join(args.out_dir, "predicted_paths_unconstrained.jsonl"),
        )

    print("\nDone. Pass the output files to run_evaluation.py:")
    print(f"  --constrained_paths   {args.out_dir}/predicted_paths_constrained.jsonl")
    if not args.skip_unconstrained:
        print(f"  --unconstrained_paths {args.out_dir}/predicted_paths_unconstrained.jsonl")


if __name__ == "__main__":
    main()
