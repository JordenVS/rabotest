"""
eval/run_evaluation.py
-----------------------
Orchestrates the full evaluation pipeline comparing four approaches:

  1. GCR (constrained)      — trie-constrained beam search (this work)
  2. Unconstrained baseline — same model, no trie ("GCR w/o constraint"
                              in the ablation study of Luo et al., 2024)
  3. GraphRAG baseline      — 1-hop subgraph context + LLM generation
                              (graphrag/graphrag.py)
  4. RAG baseline           — dense retrieval + LLM generation
                              (rag/p2prag.py)

For each approach and each eval item the script records every metric defined
in eval/metrics.py, writes per-item JSONL results, and prints a summary table.

Note on path-based metrics for RAG:
    RAG returns free-text answers, not structured reasoning paths. Metrics
    such as path_f1 and constraint_compliance will therefore be near-zero
    for RAG by design — this is the correct result for a flat retrieval
    baseline and should be reported as such in the paper.

Usage
-----
    python -m eval.run_evaluation \
        --graph         test2.graphml \
        --eval-local    eval_local.jsonl \
        --eval-obj      eval_localobj.jsonl \
        --model         Qwen/Qwen2.5-1.5B-Instruct \
        --faiss-db      ./faiss_db_pm4py \
        --embedding     bge \
        --num-paths     3 \
        --max-depth     3 \
        --out-dir       results/

All CLI arguments have sensible defaults so the script can be run with just
    python -m eval.run_evaluation
if the default file names are in the working directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any, Optional

import networkx as nx

# ---------------------------------------------------------------------------
# Resolve project root so imports work regardless of cwd
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.graph_utils import load_graphml_to_networkx
from eval.metrics import (
    score_single,
    aggregate_scores,
    _build_valid_semantic_edges,
)


# ---------------------------------------------------------------------------
# Lazy imports for heavy dependencies (avoids import errors when just reading
# the file without GPU / full env)
# ---------------------------------------------------------------------------

def _load_gcr_agent(model_id: str, graph: nx.DiGraph, device: str = "cpu"):
    from gcr.processors import GCRProcessAgent
    return GCRProcessAgent(model_id, graph, device=device)


def _load_graphrag(model_id: str):
    """Load GraphRAG search function and a pre-built LLM callable."""
    from graphrag.graphrag import perform_local_search, build_graphrag_llm
    print(f"Loading GraphRAG model: {model_id}")
    llm = build_graphrag_llm(backend="hf", model=model_id)
    return perform_local_search, llm


def _load_rag_chain(faiss_db: str, embedding_backend: str, llm_model: str):
    """Load FAISS index and build the RAG chain."""
    from rag.rag import get_retriever_from_db, create_rag_chain
    retriever = get_retriever_from_db(faiss_db, embedding_backend=embedding_backend)
    chain = create_rag_chain(
        retriever,
        llm_backend="hf",
        llm_model=llm_model,
    )
    return chain


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Per-method runners
# ---------------------------------------------------------------------------

def run_gcr(
    items: List[Dict],
    agent,
    graph: nx.DiGraph,
    valid_edges,
    num_paths: int,
    max_depth: int,
    constrained: bool,
) -> List[Dict]:
    """Run GCR (constrained) or unconstrained baseline over *items*."""
    results = []
    method = "gcr" if constrained else "unconstrained"

    for i, item in enumerate(items):
        topic_entities = item["topic_entities"]
        question = item["question"]
        seed_entity = topic_entities[0] if topic_entities else None

        if seed_entity is None:
            generated_paths = []
            timing = {"trie_build_s": 0, "generation_s": 0, "total_s": 0, "prompt_tokens": 0}
        else:
            try:
                timing = agent.timed_generate(
                    seed_entity,
                    question,
                    constrained=constrained,
                    num_paths=num_paths,
                    max_depth=max_depth,
                )
                generated_paths = timing.pop("paths")
            except Exception as e:
                print(f"  [{method}] item {item['id']} failed: {e}")
                generated_paths = []
                timing = {"trie_build_s": 0, "generation_s": 0, "total_s": 0, "prompt_tokens": 0}

        scores = score_single(generated_paths, item, graph, valid_edges)
        results.append({
            "method": method,
            **scores,
            **timing,
            "generated_paths": generated_paths,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{method}] {i + 1}/{len(items)} done")

    return results


def _graphrag_paths_from_response(response) -> List[str]:
    """
    Extract a list of path-like strings from a GraphRAG LLM response.
    The GraphRAG baseline returns free text; we wrap the whole response
    as a single 'path' string for metric computation purposes.
    """
    if response is None:
        return []
    if isinstance(response, str):
        return [response]
    # OpenAI ChatCompletion object
    try:
        content = response.choices[0].message.content
        return [content] if content else []
    except Exception:
        return [str(response)]


def run_graphrag(
    items: List[Dict],
    graph: nx.DiGraph,
    valid_edges,
    perform_local_search,
    llm,
) -> List[Dict]:
    """Run GraphRAG local-search baseline over *items*."""
    results = []

    for i, item in enumerate(items):
        topic_entities = item["topic_entities"]
        question = item["question"]
        seed_entity = topic_entities[0] if topic_entities else None

        t0 = time.perf_counter()
        if seed_entity is None:
            generated_paths = []
        else:
            try:
                response = perform_local_search(graph, seed_entity, question, llm=llm)
                generated_paths = _graphrag_paths_from_response(response)
            except Exception as e:
                print(f"  [graphrag] item {item['id']} failed: {e}")
                generated_paths = []
        elapsed = time.perf_counter() - t0

        scores = score_single(generated_paths, item, graph, valid_edges)
        results.append({
            "method": "graphrag",
            **scores,
            "trie_build_s": 0.0,
            "generation_s": elapsed,
            "total_s": elapsed,
            "prompt_tokens": 0,  # GraphRAG context length not tracked here
            "generated_paths": generated_paths,
        })

        if (i + 1) % 10 == 0:
            print(f"  [graphrag] {i + 1}/{len(items)} done")

    return results


def run_rag(
    items: List[Dict],
    chain,
    graph: nx.DiGraph,
    valid_edges,
) -> List[Dict]:
    """
    Run the dense RAG baseline over *items*.

    The chain returns {"question", "context", "answer"} where "answer" is
    free text. Following the same convention as run_graphrag, the answer is
    wrapped as a single-element path list for metric computation. Path-based
    metrics (path_f1, constraint_compliance) will therefore be near-zero by
    design and should be interpreted accordingly in the paper.
    """
    results = []

    for i, item in enumerate(items):
        question = item["question"]

        t0 = time.perf_counter()
        try:
            output = chain.invoke({"question": question})
            answer_text = output.get("answer", "")
            # Wrap free-text answer as a single path string, consistent with
            # how _graphrag_paths_from_response handles GraphRAG output.
            generated_paths = [answer_text] if answer_text else []
        except Exception as e:
            print(f"  [rag] item {item['id']} failed: {e}")
            generated_paths = []
        elapsed = time.perf_counter() - t0

        scores = score_single(generated_paths, item, graph, valid_edges)
        results.append({
            "method": "rag",
            **scores,
            "trie_build_s": 0.0,   # RAG has no trie
            "generation_s": elapsed,
            "total_s": elapsed,
            "prompt_tokens": 0,    # retrieval token count not tracked here
            "generated_paths": generated_paths,
        })

        if (i + 1) % 10 == 0:
            print(f"  [rag] {i + 1}/{len(items)} done")

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

METRIC_COLS = [
    ("hit",                  "Hit"),
    ("path_f1",              "Path-F1"),
    ("activity_accuracy",    "Act-Acc"),
    ("constraint_compliance","Compliance"),
    ("hallucination_rate",   "Halluc."),
    ("temporal_validity",    "Temp-Val"),
    ("lifecycle_coverage",   "Lifecycle"),
    ("total_s",              "Time(s)"),
]


def print_summary(all_results: List[Dict], dataset_name: str) -> None:
    """Print a LaTeX-ready summary table to stdout."""
    from collections import defaultdict

    by_method: Dict[str, List] = defaultdict(list)
    for r in all_results:
        by_method[r["method"]].append(r)

    col_width = 11
    header = f"{'Method':<20}" + "".join(f"{name:>{col_width}}" for _, name in METRIC_COLS)
    print(f"\n=== {dataset_name} ===")
    print(header)
    print("-" * len(header))

    for method, rows in by_method.items():
        agg = aggregate_scores(rows)
        row_str = f"{method:<20}"
        for key, _ in METRIC_COLS:
            val = agg.get(key, float("nan"))
            row_str += f"{val:>{col_width}.4f}"
        print(row_str)

    print()


def save_results(results: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved {len(results)} records → {path}")


# ---------------------------------------------------------------------------
# Beam-size sweep (reproduces Figure 4 of Luo et al., 2024)
# ---------------------------------------------------------------------------

def beam_sweep(
    items: List[Dict],
    agent,
    graph: nx.DiGraph,
    valid_edges,
    beam_sizes: List[int],
    max_depth: int,
    out_dir: str,
) -> None:
    """Sweep beam sizes and write one results file per size."""
    print("\n--- Beam size sweep ---")
    for k in beam_sizes:
        print(f"  K={k} ...")
        res = run_gcr(items, agent, graph, valid_edges,
                      num_paths=k, max_depth=max_depth, constrained=True)
        agg = aggregate_scores(res)
        print(
            f"    K={k}  Hit={agg['hit']:.4f}  F1={agg['path_f1']:.4f}"
            f"  Time={agg['total_s']:.2f}s"
        )
        save_results(res, os.path.join(out_dir, f"beam_k{k}_local.jsonl"))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GCR-OCEL evaluation runner")
    p.add_argument("--graph",       default="test2.graphml")
    p.add_argument("--eval-local",  default="eval_local.jsonl")
    p.add_argument("--eval-obj",    default="eval_localobj.jsonl")
    p.add_argument("--model",       default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device",      default="cpu")
    p.add_argument("--num-paths",   type=int, default=3)
    p.add_argument("--max-depth",   type=int, default=3)
    p.add_argument("--out-dir",     default="results")
    # RAG-specific arguments
    p.add_argument(
        "--faiss-db",
        default="./faiss_db_pm4py",
        help="Path to the saved FAISS index used by the RAG baseline",
    )
    p.add_argument(
        "--embedding",
        default="bge",
        choices=["openai", "bge", "minilm", "e5"],
        help="Embedding backend to use for the RAG retriever",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["gcr", "unconstrained", "graphrag", "rag"],
        choices=["gcr", "unconstrained", "graphrag", "rag"],
        help="Which methods to run",
    )
    p.add_argument(
        "--beam-sweep",
        action="store_true",
        help="Run beam-size sweep (K = 1, 3, 5, 10) on the local dataset",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load graph and datasets
    # ------------------------------------------------------------------
    print(f"Loading graph: {args.graph}")
    G = load_graphml_to_networkx(args.graph)
    valid_edges = _build_valid_semantic_edges(G)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(valid_edges)} valid semantic triples")

    datasets = {}
    for name, path in [("local", args.eval_local), ("object", args.eval_obj)]:
        if os.path.exists(path):
            datasets[name] = load_jsonl(path)
            print(f"Loaded {len(datasets[name])} {name} questions from {path}")
        else:
            print(f"WARNING: {path} not found — skipping {name} dataset")

    if not datasets:
        print("No evaluation datasets found. Run build_all_datasets() first.")
        return

    # ------------------------------------------------------------------
    # Load models / agents
    # ------------------------------------------------------------------
    agent = None
    if "gcr" in args.methods or "unconstrained" in args.methods:
        print(f"Loading GCR model: {args.model}")
        agent = _load_gcr_agent(args.model, G, device=args.device)
        print("  GCR model ready.")

    graphrag_llm = None
    graphrag_search_fn = None
    if "graphrag" in args.methods:
        graphrag_search_fn, graphrag_llm = _load_graphrag(args.model)
        print("  GraphRAG model ready.")

    rag_chain = None
    if "rag" in args.methods:
        if not os.path.exists(args.faiss_db):
            print(
                f"WARNING: FAISS index not found at '{args.faiss_db}'. "
                f"Build it first with get_retriever(...). Skipping RAG."
            )
        else:
            print(f"Loading RAG chain (embedding={args.embedding}, llm={args.model}) ...")
            rag_chain = _load_rag_chain(args.faiss_db, args.embedding, args.model)
            print("  RAG chain ready.")

    # ------------------------------------------------------------------
    # Run evaluations
    # ------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    all_results: Dict[str, List] = {name: [] for name in datasets}

    for ds_name, items in datasets.items():
        print(f"\n--- Evaluating on '{ds_name}' ({len(items)} items) ---")

        if "gcr" in args.methods and agent is not None:
            print("  Running GCR (constrained) ...")
            res = run_gcr(items, agent, G, valid_edges,
                          num_paths=args.num_paths, max_depth=args.max_depth,
                          constrained=True)
            all_results[ds_name].extend(res)
            save_results(res, os.path.join(args.out_dir, f"gcr_{ds_name}.jsonl"))

        if "unconstrained" in args.methods and agent is not None:
            print("  Running unconstrained baseline ...")
            res = run_gcr(items, agent, G, valid_edges,
                          num_paths=args.num_paths, max_depth=args.max_depth,
                          constrained=False)
            all_results[ds_name].extend(res)
            save_results(res, os.path.join(args.out_dir, f"unconstrained_{ds_name}.jsonl"))

        if "graphrag" in args.methods and graphrag_llm is not None:
            print("  Running GraphRAG baseline ...")
            res = run_graphrag(items, G, valid_edges, graphrag_search_fn, graphrag_llm)
            all_results[ds_name].extend(res)
            save_results(res, os.path.join(args.out_dir, f"graphrag_{ds_name}.jsonl"))

        if "rag" in args.methods and rag_chain is not None:
            print("  Running RAG baseline ...")
            res = run_rag(items, rag_chain, G, valid_edges)
            all_results[ds_name].extend(res)
            save_results(res, os.path.join(args.out_dir, f"rag_{ds_name}.jsonl"))

        print_summary(all_results[ds_name], ds_name)

    # ------------------------------------------------------------------
    # Optional beam-size sweep
    # ------------------------------------------------------------------
    if args.beam_sweep and agent is not None and "local" in datasets:
        beam_sweep(
            datasets["local"], agent, G, valid_edges,
            beam_sizes=[1, 3, 5, 10],
            max_depth=args.max_depth,
            out_dir=args.out_dir,
        )

    print("Evaluation complete.")


if __name__ == "__main__":
    main()