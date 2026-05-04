# eval/rerank_predicted_paths.py

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rerank.train import (
    load_reranker,
    build_path_subgraph,
    _activities_from_path_string,
)
from utils.graph_utils2 import load_graphml_to_networkx
from gcr.gcr import (
    build_events_dict_from_context_graph,
    build_event_successors_from_g_behavior,
)

from gcr.gcr import enrich_paths_with_context, reify_generated_path, build_events_dict_from_context_graph
from gcr.processors import GCRProcessAgent

import torch


def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_dataset(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(l) for l in content.splitlines() if l.strip()]


def rerank_record(
    paths: List[str],
    anchor_oid: str,
    question: str,
    context_snapshot: Dict,
    model,
    act_vocab,
    obj_vocab,
    edge_vocab,
    device: torch.device,
) -> List[str]:
    """
    Rerank a list of pre-generated beam strings using the trained GNN.
    Returns the original list unchanged if scoring fails (graceful degradation).
    """
    # Build query BoW embedding
    q_vec = torch.zeros(len(act_vocab))
    for tok in question.lower().split():
        idx = act_vocab.encode(tok)
        if idx < len(q_vec):
            q_vec[idx] += 1.0
    norm = q_vec.norm()
    if norm > 0:
        q_vec = q_vec / norm

    graphs = []
    valid_paths = []

    for path_str in paths:
        acts = _activities_from_path_string(path_str)
        if not acts:
            continue
        g = build_path_subgraph(
            acts, anchor_oid, context_snapshot,
            act_vocab, obj_vocab, edge_vocab, q_vec,
        )
        if g is None:
            continue

        try:
            from torch_geometric.data import Data
            graphs.append(g.to(device))
        except ImportError:
            graphs.append({k: v.to(device) for k, v in g.items()})

        valid_paths.append(path_str)

    if not graphs:
        return paths  # nothing could be featurised — return original order

    with torch.no_grad():
        scores = model.forward_batch(graphs).cpu().tolist()

    ranked = sorted(zip(valid_paths, scores), key=lambda x: x[1], reverse=True)
    reranked = [p for p, _ in ranked]

    # Paths that failed featurisation go to the end
    unscored = [p for p in paths if p not in valid_paths]
    return reranked + unscored


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Load reranker
    # ------------------------------------------------------------------ #
    print(f"Loading checkpoint: {args.checkpoint}")
    model, act_vocab, obj_vocab, edge_vocab = load_reranker(
        args.checkpoint, device_str=args.device
    )
    model.eval()
    device = torch.device(args.device)

    # ------------------------------------------------------------------ #
    # Optionally load graph + agent for enrichment
    # ------------------------------------------------------------------ #
    G_context        = None
    agent_events     = None  # just the events dict, no LM needed for reification

    if not args.skip_enrich:
        assert args.graph_context  is not None, "--graph_context is required for enrichment"
        assert args.graph_behavior is not None, "--graph_behavior is required for enrichment"
        assert args.model          is not None, "--model is required for enrichment"

        print(f"Loading context graph: {args.graph_context}")
        G_context = load_graphml_to_networkx(args.graph_context)

        print(f"Loading behavior graph: {args.graph_behavior}")
        G_behavior = load_graphml_to_networkx(args.graph_behavior)

        print("Building events dict…")
        agent_events = build_events_dict_from_context_graph(G_context)

        print(f"Loading tokenizer (for reification): {args.model}")
        from gcr.processors import GCRProcessAgent
        from gcr.gcr import build_event_successors_from_g_behavior
        event_successors = build_event_successors_from_g_behavior(G_behavior, agent_events)

        # We need a minimal agent-like object for reify_generated_path,
        # which uses self.events internally
        class _MinimalAgent:
            def __init__(self, events):
                self.events = events

        _agent = _MinimalAgent(agent_events)

    # ------------------------------------------------------------------ #
    # Load dataset
    # ------------------------------------------------------------------ #
    print(f"Loading dataset: {args.dataset}")
    questions = load_dataset(args.dataset)
    instance_meta = {
        q["instance_id"]: {
            "context_snapshot": q.get("context_snapshot", {"nodes": [], "edges": []}),
            "question":         q["question"],
            "anchor_oid":       q["anchor_object"]["oid"],
        }
        for q in questions
    }

    print(f"Loading predicted paths: {args.paths}")
    records = load_jsonl(args.paths)
    print(f"  {len(records)} records loaded.\n")

    # ------------------------------------------------------------------ #
    # Rerank (+ optionally enrich)
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f_out:
        for rec in tqdm(records, desc="Reranking"):
            instance_id = rec["instance_id"]
            paths       = rec.get("paths", [])
            meta        = instance_meta.get(instance_id)

            if not paths or meta is None:
                f_out.write(json.dumps(rec, default=str) + "\n")
                continue

            # --- Rerank ---
            reranked_paths = rerank_record(
                paths=paths,
                anchor_oid=meta["anchor_oid"],
                question=meta["question"],
                context_snapshot=meta["context_snapshot"],
                model=model,
                act_vocab=act_vocab,
                obj_vocab=obj_vocab,
                edge_vocab=edge_vocab,
                device=device,
            )

            # --- Enrich top-k of the reranked paths ---
            context_block = None
            if not args.skip_enrich and G_context is not None:
                reified_paths = [
                    reify_generated_path(
                        _agent,
                        generated_string=p,
                        anchor_object=meta["anchor_oid"],
                        G_context=G_context,
                    )
                    for p in reranked_paths[:args.enrich_top_k]
                ]
                context_block = enrich_paths_with_context(
                    paths=reified_paths,
                    anchor_object=meta["anchor_oid"],
                    G_context=G_context,
                )

            out_rec = {
                **rec,
                "paths":         reranked_paths,
                "context_block": context_block,
            }
            f_out.write(json.dumps(out_rec, default=str) + "\n")

    print(f"\nReranked paths → {args.out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Rerank pre-generated GCR beam paths using the trained GNN reranker. "
            "Reads an existing predicted_paths_constrained.jsonl and writes a "
            "new JSONL with paths reordered by GNN relevance score."
        )
    )
    p.add_argument("--paths",      required=True,
                   help="predicted_paths_constrained.jsonl from generate_predicted_paths.py")
    p.add_argument("--dataset",    required=True,
                   help="sampled_100.json — needed for context_snapshot and question text")
    p.add_argument("--checkpoint", required=True,
                   help="Trained GNN checkpoint (.pt) from rerank/train.py")
    p.add_argument("--out",        default="results/predicted_paths_reranked.jsonl",
                   help="Output path for the reranked JSONL")
    p.add_argument("--graph_context",  default="graphs/context_graph.graphml",
               help="Context graph GraphML — required for enrichment.")
    p.add_argument("--graph_behavior", default="graphs/behavior_graph.graphml",
                help="Behavior graph GraphML — required for enrichment.")
    p.add_argument("--model",          default="Qwen/Qwen2.5-1.5B-Instruct",
                help="HuggingFace model ID — required for reification (tokenizer only).")
    p.add_argument("--enrich_top_k",   type=int, default=1,
                help="Enrich only the top-k reranked paths (default: 1).")
    p.add_argument("--skip_enrich",    action="store_true",
               help="Skip enrichment entirely — just reorder paths.")
    p.add_argument("--device",     default="cpu",
                   choices=["cpu", "cuda", "mps"])
    return p.parse_args()


if __name__ == "__main__":
    main()