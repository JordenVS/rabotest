"""
gnn_reranker.py
================
Graph Neural Network reranker for PA-GCR beam paths.

Architecture overview
---------------------
Each candidate path is represented as a small query-conditioned subgraph.
A Graph Attention Network (GAT) reads this subgraph and projects the
graph-level embedding to a scalar relevance score.

Path generation strategy
------------------------
Running the full PA-GCR beam search (load LM, build trie, decode) over
tens of thousands of training instances and again over the evaluation set
is computationally intractable.  Instead, both training and inference
generate candidate paths via *object-valid random walks* over G_behavior.

A walk π = (e₁, …, eₖ) anchored at object o* is object-valid iff:
  1. Anchor participation:  o* ∈ e₁.objects
  2. Behavioral adjacency:  (eᵢ, eᵢ₊₁) ∈ E_behavior  for all i
     — guaranteed by sampling only from event_successors, which is built
       exclusively from behavior edges in G_behavior.
  3. Object continuity:     eᵢ.objects ∩ active_objects ≠ ∅  for all i

These are the same three constraints enforced by the Process Trie during
constrained beam search (Section 3.3 of the thesis).  The walk replaces
LM logit scoring with uniform random selection, making path generation
O(k · D) per instance instead of requiring a forward pass through the LM.

Both training and inference call generate_walks(), which uses the Event
objects and event_successors map produced by the project's existing
utilities:
  - build_events_dict_from_context_graph  (gcr.gcr)
  - build_event_successors_from_g_behavior (gcr.gcr)
  - load_graphml_to_networkx              (utils.graph_utils2)

Training signal
---------------
Pairwise margin ranking loss (Burges et al., 2005):
    loss = max(0, margin − score(positive) + score(negative))
A walk is *positive* if its activity sequence contains the gold_answer
token after normalisation; *negative* otherwise.

References
----------
- Veličković et al., 2018. Graph Attention Networks. ICLR 2018.
- Luo et al., 2025. Graph-Constrained Reasoning. (GCR framework)
- Hamilton et al., 2017. Inductive Representation Learning on Large Graphs.
- Burges et al., 2005. Learning to Rank using Gradient Descent.
- van der Aalst, 2023. Object-Centric Process Mining. (OCEL standard)
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports — Event objects and graph builders
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gcr.event import Event
from gcr.gcr import (
    build_events_dict_from_context_graph,
    build_event_successors_from_g_behavior,
)
from rerank.pathrerankergnn import PathRerankerGNN
from rerank.pathrerankerdataset import PathRerankingDataset
from utils.graph_utils2 import load_graphml_to_networkx

# ---------------------------------------------------------------------------
# Optional: PyTorch Geometric for multi-head GAT + batched pooling.
# Falls back to a hand-written single-head GAT if PyG is not installed.
# ---------------------------------------------------------------------------
try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    _USE_PYG = True
except ImportError:
    _USE_PYG = False


# ===========================================================================
# 1.  Vocabulary
# ===========================================================================

class VocabEncoder:
    """Maps string tokens to integer indices; built from the training corpus."""

    def __init__(self, pad: str = "<PAD>", unk: str = "<UNK>"):
        self.pad = pad
        self.unk = unk
        self._tok2idx: Dict[str, int] = {pad: 0, unk: 1}

    def fit(self, tokens: List[str]) -> "VocabEncoder":
        for t in tokens:
            if t not in self._tok2idx:
                self._tok2idx[t] = len(self._tok2idx)
        return self

    def encode(self, token: str) -> int:
        return self._tok2idx.get(token, self._tok2idx[self.unk])

    def __len__(self) -> int:
        return len(self._tok2idx)


def _normalise_activity(act: str) -> str:
    """Lowercase; strip the 'Event:' prefix and underscores used in path strings."""
    return act.lower().replace("event:", "").replace("_", " ").strip()


def _activities_from_path_string(path_str: str) -> List[str]:
    """'Event:Create_PO Event:Approve_PO' → ['create po', 'approve po']"""
    return [_normalise_activity(tok) for tok in path_str.split() if tok.strip()]


# ===========================================================================
# 2.  Object-valid random walk over G_behavior
# ===========================================================================

def _anchor_start_events(
    anchor_oid: str,
    events:     Dict[str, Event],
) -> List[Event]:
    """
    Return all Event objects in which anchor_oid participates, sorted
    chronologically by Event.timestamp.  These are the valid walk seeds
    satisfying constraint 1 (anchor participation).
    """
    candidates = [e for e in events.values() if anchor_oid in e.objects]
    return sorted(candidates, key=lambda e: str(e.timestamp or ""))


def _random_walk(
    start:            Event,
    event_successors: Dict[str, List[Event]],
    max_depth:        int,
    rng:              random.Random,
) -> List[Event]:
    """
    Perform a single constrained random walk starting from *start*.

    event_successors is the dict produced by build_event_successors_from_g_behavior:
      keys   = event IDs (str)
      values = List[Event] of successor Event objects reachable via
               G_behavior edges with edge_type == 'behavior'

    At each hop we collect successors (behavioral adjacency, constraint 2)
    whose .objects set intersects the accumulated active-object set (object
    continuity, constraint 3), then sample uniformly.

    Parameters
    ----------
    start            : Seed Event satisfying anchor participation.
    event_successors : Dict[eid → List[Event]] from G_behavior.
    max_depth        : Maximum number of events in the walk.
    rng              : Per-instance seeded random.Random.

    Returns
    -------
    List[Event] of length 1 … max_depth.
    """
    path        = [start]
    active_objs = set(start.objects)

    for _ in range(max_depth - 1):
        # Successors are Event objects; filter by object continuity
        valid = [
            e for e in event_successors.get(path[-1].eid, [])
            if e.objects & active_objs
        ]
        if not valid:
            break
        nxt = rng.choice(valid)
        path.append(nxt)
        active_objs |= nxt.objects

    return path


def _walk_to_path_string(walk: List[Event]) -> str:
    """
    Convert a list of Event objects to the canonical path string used
    throughout the PA-GCR pipeline:
        'Event:Create_Purchase_Order Event:Approve_Purchase_Order …'
    Spaces in activity names are replaced by underscores to preserve
    single-token atomicity (Section 4.2.2 of the thesis).
    """
    return " ".join("Event:" + e.activity.replace(" ", "_") for e in walk)


def generate_walks(
    anchor_oid:       str,
    events:           Dict[str, Event],
    event_successors: Dict[str, List[Event]],
    num_paths:        int,
    max_depth:        int,
    rng:              random.Random,
) -> List[str]:
    """
    Generate up to *num_paths* unique object-valid walk strings for one
    anchor object.  Attempts up to num_paths × 10 walks, cycling through
    all valid start events, and deduplicates at the path-string level.
    Returns fewer paths when the anchor's local graph is sparse.

    Parameters
    ----------
    anchor_oid       : Anchor object identifier (e.g. 'purchase_order:42').
    events           : Full events dict from build_events_dict_from_context_graph.
    event_successors : Successor map from build_event_successors_from_g_behavior.
    num_paths        : Desired number of distinct walk strings.
    max_depth        : Maximum walk length in events.
    rng              : Seeded random.Random instance.
    """
    starts = _anchor_start_events(anchor_oid, events)
    if not starts:
        return []

    seen:  Set[str] = set()
    paths: List[str] = []

    for attempt in range(num_paths * 10):
        if len(paths) >= num_paths:
            break
        start    = starts[attempt % len(starts)]
        walk     = _random_walk(start, event_successors, max_depth, rng)
        path_str = _walk_to_path_string(walk)
        if path_str and path_str not in seen:
            seen.add(path_str)
            paths.append(path_str)

    return paths


# ===========================================================================
# 3.  Vocabulary fitting (seeded from the behavior graph, not beam files)
# ===========================================================================

def fit_vocabularies(
    instances:        List[Dict],
    events:           Dict[str, Event],
    event_successors: Dict[str, List[Event]],
) -> Tuple[VocabEncoder, VocabEncoder, VocabEncoder]:
    """
    Build activity, object-type, and edge-label vocabularies entirely from
    the behavior graph and instance context snapshots — no pre-computed
    beam files required.

    The activity vocabulary is seeded from every Event in the events dict
    (the full G_behavior node set) so that walk-generated activities at
    inference time are never out-of-vocabulary, even for eval instances
    whose anchor objects were not seen during training.

    Parameters
    ----------
    instances        : Training or eval instances; used for object-type
                       and edge-label vocabularies from context snapshots.
    events           : Full events dict (covers all activities in the log).
    event_successors : Included for API symmetry; activities already covered
                       via the events dict.
    """
    act_vocab  = VocabEncoder()
    obj_vocab  = VocabEncoder()
    edge_vocab = VocabEncoder()

    # Activities: every event node in G_behavior (via events dict)
    all_activities: List[str] = [
        _normalise_activity(e.activity) for e in events.values()
    ]
    # Also include gold-path activities from instances for extra coverage
    for inst in instances:
        for path_set in inst.get("gold_paths", []):
            for seq in (path_set if path_set and isinstance(path_set[0], list)
                        else [path_set]):
                for act in seq:
                    all_activities.append(_normalise_activity(str(act)))

    # Object types and edge labels: from context snapshots
    all_obj_types:   List[str] = []
    all_edge_labels: List[str] = ["path_sequence", "unknown"]

    for inst in instances:
        snap = inst.get("context_snapshot", {})
        for node in snap.get("nodes", []):
            obj_type = node.get("object_type", node.get("type", "unknown"))
            all_obj_types.append(str(obj_type))
        for edge in snap.get("edges", []):
            all_edge_labels.append(
                str(edge.get("label", edge.get("edge_type", "unknown")))
            )

    act_vocab.fit(all_activities)
    obj_vocab.fit(all_obj_types)
    edge_vocab.fit(all_edge_labels)

    print(
        f"Vocabularies fitted — "
        f"activities: {len(act_vocab)}, "
        f"object types: {len(obj_vocab)}, "
        f"edge labels: {len(edge_vocab)}."
    )
    return act_vocab, obj_vocab, edge_vocab


# ===========================================================================
# 4.  Subgraph featurisation
# ===========================================================================

def build_path_subgraph(
    path_activities:  List[str],
    anchor_oid:       str,
    context_snapshot: Dict,
    act_vocab:        VocabEncoder,
    obj_vocab:        VocabEncoder,
    edge_vocab:       VocabEncoder,
    query_embedding:  torch.Tensor,
) -> Optional[object]:
    """
    Construct a PyG Data object (or plain dict fallback) representing the
    ego-subgraph for one (path, query) pair.

    Nodes
    -----
    - One virtual node per distinct activity in the path  (event type)
    - One node per entry in context_snapshot["nodes"]     (object type)

    Node features  [act_onehot | obj_onehot | query_sim | is_anchor]
    - Activity nodes: one-hot over act_vocab + cosine-sim against BoW query
    - Object nodes:   one-hot over obj_vocab

    Edges
    -----
    - Sequential path_sequence edges between consecutive activity nodes
      (directed + reverse for undirected message passing)
    - Edges from context_snapshot labelled by their label/edge_type field

    Returns None when the snapshot is empty (degenerate instance).
    """
    snap_nodes = {n["id"]: n for n in context_snapshot.get("nodes", [])}
    snap_edges = context_snapshot.get("edges", [])

    if not snap_nodes:
        return None

    node_ids:       List[str]      = []
    node_id_to_idx: Dict[str, int] = {}

    # Path activity nodes — virtual, identified by the __act__ prefix
    for act in path_activities:
        nid = f"__act__{act}"
        if nid not in node_id_to_idx:
            node_id_to_idx[nid] = len(node_ids)
            node_ids.append(nid)

    # Context snapshot nodes (objects / events from the ego-graph)
    for nid in snap_nodes:
        if nid not in node_id_to_idx:
            node_id_to_idx[nid] = len(node_ids)
            node_ids.append(nid)

    n_nodes  = len(node_ids)
    act_dim  = len(act_vocab)
    obj_dim  = len(obj_vocab)
    feat_dim = act_dim + obj_dim + 2   # +query_sim, +is_anchor

    x = torch.zeros(n_nodes, feat_dim)

    for i, nid in enumerate(node_ids):
        x[i, feat_dim - 1] = float(nid == anchor_oid)   # is_anchor flag

        if nid.startswith("__act__"):
            act_name = nid[7:]   # strip '__act__' prefix
            act_idx  = act_vocab.encode(act_name)
            if act_idx < act_dim:
                x[i, act_idx] = 1.0
            # Query similarity: dot of one-hot with BoW query vector
            if query_embedding is not None and act_idx < query_embedding.shape[0]:
                x[i, feat_dim - 2] = float(query_embedding[act_idx])
        else:
            node_data = snap_nodes.get(nid, {})
            obj_type  = node_data.get("object_type", node_data.get("type", "unknown"))
            obj_idx   = obj_vocab.encode(str(obj_type))
            if obj_idx < obj_dim:
                x[i, act_dim + obj_idx] = 1.0

    src_list:   List[int] = []
    dst_list:   List[int] = []
    eattr_list: List[int] = []

    path_seq_label = edge_vocab.encode("path_sequence")

    # Sequential edges along the path reasoning chain
    for k in range(len(path_activities) - 1):
        s = node_id_to_idx[f"__act__{path_activities[k]}"]
        d = node_id_to_idx[f"__act__{path_activities[k + 1]}"]
        src_list   += [s, d]
        dst_list   += [d, s]
        eattr_list += [path_seq_label, path_seq_label]

    # Context snapshot edges
    for edge in snap_edges:
        s_id = str(edge.get("source", ""))
        d_id = str(edge.get("target", ""))
        if s_id in node_id_to_idx and d_id in node_id_to_idx:
            s      = node_id_to_idx[s_id]
            d      = node_id_to_idx[d_id]
            elabel = edge_vocab.encode(
                str(edge.get("label", edge.get("edge_type", "unknown")))
            )
            src_list   += [s, d]
            dst_list   += [d, s]
            eattr_list += [elabel, elabel]

    if not src_list:
        # Degenerate: add self-loop on anchor to avoid empty edge set
        anchor_idx = node_id_to_idx.get(anchor_oid, 0)
        src_list   = [anchor_idx]
        dst_list   = [anchor_idx]
        eattr_list = [0]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(eattr_list,           dtype=torch.long)

    if _USE_PYG:
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=n_nodes)
    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}




# ===========================================================================
# 7.  Training loop
# ===========================================================================

def train(
    train_instances:  List[Dict],
    events:           Dict[str, Event],
    event_successors: Dict[str, List[Event]],
    *,
    num_paths:        int   = 5,
    max_depth:        int   = 7,
    hidden_dim:       int   = 128,
    n_heads:          int   = 4,
    dropout:          float = 0.1,
    margin:           float = 0.5,
    lr:               float = 3e-4,
    weight_decay:     float = 1e-4,
    n_epochs:         int   = 20,
    batch_size:       int   = 32,
    seed:             int   = 42,
    device_str:       str   = "cpu",
    checkpoint_path:  str   = "gnn_reranker.pt",
) -> PathRerankerGNN:
    """
    Train the GNN reranker using random-walk-derived training pairs.

    Parameters
    ----------
    train_instances  : Instances from train_dataset.json.  Anchor objects
                       must be disjoint from sampled_100 (enforced by
                       generate_eval_dataset.py --existing_sample).
    events           : Full events dict from build_events_dict_from_context_graph.
    event_successors : Successor map from build_event_successors_from_g_behavior.
    num_paths        : Random walks generated per training instance.
    max_depth        : Maximum walk length in events.
    All other kwargs : Standard optimiser / architecture hyperparameters.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_str)

    act_vocab, obj_vocab, edge_vocab = fit_vocabularies(
        train_instances, events, event_successors
    )

    dataset = PathRerankingDataset(
        train_instances, events, event_successors,
        act_vocab, obj_vocab, edge_vocab,
        num_paths=num_paths, max_depth=max_depth, seed=seed,
    )

    print("Materialising training pairs via random walks…")
    pairs = dataset.get_pairs()
    print(f"Training pairs (pos, neg): {len(pairs)}")

    if not pairs:
        raise ValueError(
            "No training pairs generated.  Verify that G_behavior contains "
            "edges with edge_type='behavior' and that anchor objects "
            "participate in events with valid successors."
        )

    random.shuffle(pairs)

    feat_dim     = len(act_vocab) + len(obj_vocab) + 2
    n_edge_types = len(edge_vocab)

    model = PathRerankerGNN(
        feat_dim=feat_dim, hidden_dim=hidden_dim,
        n_edge_types=n_edge_types, n_heads=n_heads, dropout=dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    best_loss  = float("inf")

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for start in tqdm(range(0, len(pairs), batch_size),
                          desc=f"Epoch {epoch}/{n_epochs}", leave=False):
            batch_meta = pairs[start : start + batch_size]
            pos_graphs, neg_graphs = [], []

            for meta in batch_meta:
                pg, ng = dataset.build_graphs(meta)
                if pg is None or ng is None:
                    continue
                if _USE_PYG:
                    pos_graphs.append(pg.to(device))
                    neg_graphs.append(ng.to(device))
                else:
                    pos_graphs.append({k: v.to(device) for k, v in pg.items()})
                    neg_graphs.append({k: v.to(device) for k, v in ng.items()})

            if not pos_graphs:
                continue

            pos_scores = model.forward_batch(pos_graphs)
            neg_scores = model.forward_batch(neg_graphs)
            # Pairwise hinge loss: penalise whenever neg is scored above pos
            loss = F.relu(margin - pos_scores + neg_scores).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch:3d} | loss {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "act_vocab":    act_vocab._tok2idx,
                "obj_vocab":    obj_vocab._tok2idx,
                "edge_vocab":   edge_vocab._tok2idx,
                "feat_dim":     feat_dim,
                "hidden_dim":   hidden_dim,
                "n_edge_types": n_edge_types,
                "n_heads":      n_heads,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved (loss {best_loss:.4f})")

    model.eval()
    return model


# ===========================================================================
# 8.  Inference — generate walks on-the-fly, then rerank
# ===========================================================================

def load_reranker(
    checkpoint_path: str,
    device_str:      str = "cpu",
) -> Tuple[PathRerankerGNN, VocabEncoder, VocabEncoder, VocabEncoder]:
    """Load a trained reranker and its frozen vocabularies from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device_str)

    act_vocab  = VocabEncoder(); act_vocab._tok2idx  = ckpt["act_vocab"]
    obj_vocab  = VocabEncoder(); obj_vocab._tok2idx  = ckpt["obj_vocab"]
    edge_vocab = VocabEncoder(); edge_vocab._tok2idx = ckpt["edge_vocab"]

    model = PathRerankerGNN(
        feat_dim=ckpt["feat_dim"],
        hidden_dim=ckpt["hidden_dim"],
        n_edge_types=ckpt["n_edge_types"],
        n_heads=ckpt["n_heads"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device_str).eval()
    return model, act_vocab, obj_vocab, edge_vocab


def rerank_walks(
    model:            PathRerankerGNN,
    anchor_oid:       str,
    question:         str,
    context_snapshot: Dict,
    events:           Dict[str, Event],
    event_successors: Dict[str, List[Event]],
    act_vocab:        VocabEncoder,
    obj_vocab:        VocabEncoder,
    edge_vocab:       VocabEncoder,
    num_paths:        int = 5,
    max_depth:        int = 7,
    seed:             int = 0,
    device_str:       str = "cpu",
) -> List[Tuple[str, float]]:
    """
    Generate object-valid random walks for one instance then rank them by
    GNN score.  No pre-computed beam files are required.

    Parameters
    ----------
    model            : Trained PathRerankerGNN (eval mode).
    anchor_oid       : Anchor object identifier.
    question         : Natural-language query string.
    context_snapshot : Instance context snapshot dict.
    events           : Full events dict (from build_events_dict_from_context_graph).
    event_successors : Successor map (from build_event_successors_from_g_behavior).
    act_vocab        : Fitted activity VocabEncoder (from checkpoint).
    obj_vocab        : Fitted object-type VocabEncoder (from checkpoint).
    edge_vocab       : Fitted edge-label VocabEncoder (from checkpoint).
    num_paths        : Number of walk candidates to generate and score.
    max_depth        : Maximum walk length in events.
    seed             : Random seed (use instance index for reproducibility).
    device_str       : Torch device string.

    Returns
    -------
    List of (path_string, score) sorted by score descending.
    """
    device = torch.device(device_str)
    rng    = random.Random(seed)

    walk_strings = generate_walks(
        anchor_oid, events, event_successors, num_paths, max_depth, rng
    )
    if not walk_strings:
        return []

    # Build query BoW embedding (same construction as during training)
    q_vec = torch.zeros(len(act_vocab))
    for tok in question.lower().split():
        idx = act_vocab.encode(tok)
        if idx < len(q_vec):
            q_vec[idx] += 1.0
    norm = q_vec.norm()
    if norm > 0:
        q_vec = q_vec / norm

    graphs:      List[object] = []
    valid_walks: List[str]    = []

    for ws in walk_strings:
        acts = _activities_from_path_string(ws)
        if not acts:
            continue
        g = build_path_subgraph(
            acts, anchor_oid, context_snapshot,
            act_vocab, obj_vocab, edge_vocab, q_vec,
        )
        if g is None:
            continue
        graphs.append(
            g.to(device) if _USE_PYG else {k: v.to(device) for k, v in g.items()}
        )
        valid_walks.append(ws)

    if not graphs:
        return [(ws, 0.0) for ws in walk_strings]

    with torch.no_grad():
        scores = model.forward_batch(graphs).cpu().tolist()

    return sorted(zip(valid_walks, scores), key=lambda x: x[1], reverse=True)


# ===========================================================================
# 9.  CLI
# ===========================================================================

def _load_instances(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(l) for l in content.splitlines() if l.strip()]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Train or run the GNN path reranker with object-valid random walks."
    )
    p.add_argument("--mode", choices=["train", "rerank"], default="train")
    # Graph inputs — required for both modes
    p.add_argument("--graph_context",  required=True,
                   help="Context graph GraphML (Event + Object nodes, participation edges).")
    p.add_argument("--graph_behavior", required=True,
                   help="Behavior graph GraphML (Event nodes only, behavior edges).")
    # Dataset inputs
    p.add_argument("--train_dataset",  required=False,
                   help="train_dataset.json produced by generate_eval_dataset.py.")
    p.add_argument("--eval_dataset",   required=False,
                   help="sampled_100.json for inference demo.")
    # Walk hyperparameters
    p.add_argument("--num_paths",      type=int,   default=5,
                   help="Random walks per instance (default: 5).")
    p.add_argument("--max_depth",      type=int,   default=7,
                   help="Maximum walk length in events (default: 7).")
    # Model / training hyperparameters
    p.add_argument("--checkpoint",     default="gnn_reranker.pt")
    p.add_argument("--hidden_dim",     type=int,   default=128)
    p.add_argument("--n_heads",        type=int,   default=4)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--margin",         type=float, default=0.5)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--n_epochs",       type=int,   default=20)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         default="cpu")
    args = p.parse_args()

    # ------------------------------------------------------------------ #
    # Load graphs and build derived structures (shared by both modes)
    # ------------------------------------------------------------------ #
    print(f"Loading context graph:  {args.graph_context}")
    G_context = load_graphml_to_networkx(args.graph_context)

    print(f"Loading behavior graph: {args.graph_behavior}")
    G_behavior = load_graphml_to_networkx(args.graph_behavior)

    print("Building events dict from context graph…")
    events = build_events_dict_from_context_graph(G_context)
    print(f"  {len(events)} events loaded.")

    print("Building event successors from behavior graph…")
    event_successors = build_event_successors_from_g_behavior(G_behavior, events)
    print(f"  {len(event_successors)} events have successors.\n")

    # ------------------------------------------------------------------ #
    if args.mode == "train":
        assert args.train_dataset, "--train_dataset is required for training."
        train_instances = _load_instances(args.train_dataset)
        print(f"Loaded {len(train_instances)} training instances.\n")

        train(
            train_instances, events, event_successors,
            num_paths=args.num_paths,
            max_depth=args.max_depth,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            dropout=args.dropout,
            margin=args.margin,
            lr=args.lr,
            weight_decay=args.weight_decay,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device_str=args.device,
            checkpoint_path=args.checkpoint,
        )

    elif args.mode == "rerank":
        assert args.eval_dataset, "--eval_dataset is required for reranking."
        model, act_vocab, obj_vocab, edge_vocab = load_reranker(
            args.checkpoint, device_str=args.device
        )
        eval_instances = _load_instances(args.eval_dataset)

        for inst_idx, inst in enumerate(eval_instances[:5]):   # demo: first 5
            ranked = rerank_walks(
                model,
                anchor_oid=inst["anchor_object"]["oid"],
                question=inst["question"],
                context_snapshot=inst.get("context_snapshot",
                                          {"nodes": [], "edges": []}),
                events=events,
                event_successors=event_successors,
                act_vocab=act_vocab,
                obj_vocab=obj_vocab,
                edge_vocab=edge_vocab,
                num_paths=args.num_paths,
                max_depth=args.max_depth,
                seed=args.seed + inst_idx,
                device_str=args.device,
            )
            print(f"\n[{inst['instance_id']}] {inst['question']}")
            print(f"  Gold: {inst['gold_answer']}")
            for rank, (path, score) in enumerate(ranked, 1):
                print(f"  #{rank} (score={score:.3f}): {path[:100]}")