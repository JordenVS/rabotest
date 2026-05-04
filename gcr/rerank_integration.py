# gcr/rerank_integration.py

from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple
import torch

from rerank.train import (
    load_reranker,
    rerank_walks,
    _activities_from_path_string,
    VocabEncoder,
)
from rerank.pathrerankergnn import PathRerankerGNN
from gcr.event import Event


def rerank_beam_paths(
    paths: List[str],
    anchor_oid: str,
    question: str,
    context_snapshot: Dict,
    events: Dict[str, Event],
    event_successors: Dict[str, List[Event]],
    model: PathRerankerGNN,
    act_vocab: VocabEncoder,
    obj_vocab: VocabEncoder,
    edge_vocab: VocabEncoder,
    device_str: str = "cpu",
    seed: int = 0,
) -> List[str]:
    """
    Rerank a list of decoded beam strings using the trained GNN reranker.

    The reranker scores each path as a subgraph and returns them sorted by
    relevance score (descending). The beam strings are the direct output of
    GCRProcessProcessor and have the canonical format:
        'Event:Create_Purchase_Order Event:Approve_Purchase_Order ...'

    This function converts beam strings to activity lists, builds subgraphs,
    scores them, and returns the reordered strings — preserving the original
    string format expected by enrich_paths_with_context and the LLM prompt.

    Parameters
    ----------
    paths           : Decoded beam strings from constrained generation.
    anchor_oid      : Anchor object identifier for this instance.
    question        : Natural-language question (used for query embedding).
    context_snapshot: Instance context snapshot from the eval dataset.
    events          : Full events dict.
    event_successors: Behavior successor map.
    model           : Trained PathRerankerGNN in eval mode.
    act/obj/edge_vocab : Frozen vocabularies from the checkpoint.
    device_str      : Torch device.
    seed            : Random seed (use instance index for reproducibility).

    Returns
    -------
    List[str] — beam strings reordered by GNN relevance score (best first).
    """
    from rerank.train import build_path_subgraph, _activities_from_path_string

    device = torch.device(device_str)

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

        # Handle both PyG and fallback dict graphs
        try:
            from torch_geometric.data import Data
            graphs.append(g.to(device))
        except ImportError:
            graphs.append({k: v.to(device) for k, v in g.items()})

        valid_paths.append(path_str)

    if not graphs:
        return paths  # graceful degradation: return original order

    with torch.no_grad():
        scores = model.forward_batch(graphs).cpu().tolist()

    ranked = sorted(zip(valid_paths, scores), key=lambda x: x[1], reverse=True)
    reranked_paths = [p for p, _ in ranked]

    # Append any paths that couldn't be featurised at the end
    unscored = [p for p in paths if p not in valid_paths]
    return reranked_paths + unscored