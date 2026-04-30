"""
gnn_reranker.py
================
Graph Neural Network reranker for PA-GCR beam paths.

Architecture overview
---------------------
Each candidate path produced by the constrained beam search is a
sequence of events grounded in G_context.  The reranker scores every
(query, path) pair by building a small query-conditioned subgraph and
running a Graph Attention Network (GAT) over it, then projecting the
graph-level embedding to a scalar relevance score.

The training signal is a pairwise margin ranking loss constructed from
the labelled training data:

    loss = max(0, margin - score(positive_path) + score(negative_path))

where a *positive* path is one whose reified event sequence contains the
gold_answer activity, and a *negative* path is any other beam for the
same query.

References
----------
- Veličković et al., 2018. Graph Attention Networks. ICLR 2018.
- Luo et al., 2025. Graph-Constrained Reasoning. (GCR framework)
- van der Aalst, 2023. Object-Centric Process Mining. (OCEL standard)
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Optional: PyG (PyTorch Geometric) for GAT layers.
# Falls back to a hand-written message-passing GAT if PyG is not available.
# ---------------------------------------------------------------------------
try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    _USE_PYG = True
except ImportError:
    _USE_PYG = False


# ===========================================================================
# 1.  Featurisation
# ===========================================================================

class VocabEncoder:
    """
    Maps string tokens (activity names, object types, edge labels) to
    integer indices.  Built lazily from the training corpus.
    """

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
    """Lowercase and strip the 'Event:' prefix that appears in path strings."""
    return act.lower().replace("event:", "").replace("_", " ").strip()


def path_string_to_activities(path_str: str) -> List[str]:
    """
    Convert a raw beam path string like
        'Event:Create_Purchase_Order Event:Approve_Purchase_Order'
    to a list of normalised activity names.
    """
    return [_normalise_activity(tok) for tok in path_str.split()
            if tok.strip()]


# ---------------------------------------------------------------------------
# Subgraph construction
# ---------------------------------------------------------------------------

def build_path_subgraph(
    path_activities: List[str],
    anchor_oid: str,
    context_snapshot: Dict,
    act_vocab: VocabEncoder,
    obj_vocab: VocabEncoder,
    edge_vocab: VocabEncoder,
    query_embedding: torch.Tensor,          # shape (D_query,)
) -> Optional["Data"]:
    """
    Construct a PyG Data object (or equivalent dict) representing the
    ego-subgraph around the path, augmented with a query-conditioned
    node feature.

    Node feature vector
    -------------------
    For each node in the subgraph we concatenate:
      [activity_onehot | obj_type_onehot | query_sim_scalar | is_anchor]

    Activity nodes (events in the path) receive an activity feature;
    object nodes receive an object-type feature.  The query similarity
    scalar is the cosine similarity between a fixed bag-of-words query
    embedding and a per-node bag-of-words activity embedding — this
    provides a lightweight relevance signal without requiring a full LM.

    Edge feature
    ------------
    Edge type (lifecycle, participation, structure) encoded as an
    integer index.

    Parameters
    ----------
    path_activities   : Ordered list of activity names for this beam.
    anchor_oid        : The anchor object identifier (str).
    context_snapshot  : ``instance["context_snapshot"]`` dict with
                        ``nodes`` and ``edges`` lists.
    act_vocab         : Fitted activity vocabulary encoder.
    obj_vocab         : Fitted object-type vocabulary encoder.
    edge_vocab        : Fitted edge-label vocabulary encoder.
    query_embedding   : Pre-computed query embedding (bag-of-words or
                        sentence-transformer vector).

    Returns
    -------
    PyG Data object (if PyG available) or dict with keys
    ``x``, ``edge_index``, ``edge_attr``.  Returns None if the subgraph
    is degenerate (zero nodes).
    """
    snap_nodes = {n["id"]: n for n in context_snapshot.get("nodes", [])}
    snap_edges = context_snapshot.get("edges", [])

    if not snap_nodes:
        return None

    # Build node list: path events first, then snapshot objects
    node_ids: List[str] = []
    node_id_to_idx: Dict[str, int] = {}

    # Add path activities as virtual "event" nodes
    for act in path_activities:
        nid = f"__act__{act}"
        if nid not in node_id_to_idx:
            node_id_to_idx[nid] = len(node_ids)
            node_ids.append(nid)

    # Add context snapshot nodes (objects + events from ego-graph)
    for nid in snap_nodes:
        if nid not in node_id_to_idx:
            node_id_to_idx[nid] = len(node_ids)
            node_ids.append(nid)

    n_nodes = len(node_ids)
    act_dim = len(act_vocab)
    obj_dim = len(obj_vocab)
    feat_dim = act_dim + obj_dim + 2  # +2: query_sim, is_anchor

    x = torch.zeros(n_nodes, feat_dim)

    for i, nid in enumerate(node_ids):
        is_anchor = float(nid == anchor_oid)
        x[i, feat_dim - 1] = is_anchor

        if nid.startswith("__act__"):
            act_name = nid[7:]  # strip '__act__'
            act_idx = act_vocab.encode(act_name)
            if act_idx < act_dim:
                x[i, act_idx] = 1.0
            # query similarity: dot product of one-hot with query embedding
            # (a proper model would use sentence embeddings; this is a
            #  lightweight stand-in that works without an external LM)
            if query_embedding is not None and act_idx < query_embedding.shape[0]:
                x[i, feat_dim - 2] = float(query_embedding[act_idx])
        else:
            node_data = snap_nodes.get(nid, {})
            obj_type = node_data.get("ocel:type", node_data.get("type", "unknown"))
            obj_idx = obj_vocab.encode(str(obj_type))
            if obj_idx < obj_dim:
                x[i, act_dim + obj_idx] = 1.0

    # Sequential edges along the path (the primary reasoning chain)
    src_list, dst_list, eattr_list = [], [], []
    path_event_chain_label = edge_vocab.encode("path_sequence")

    for k in range(len(path_activities) - 1):
        s = node_id_to_idx[f"__act__{path_activities[k]}"]
        d = node_id_to_idx[f"__act__{path_activities[k+1]}"]
        src_list.append(s); dst_list.append(d)
        src_list.append(d); dst_list.append(s)  # undirected
        eattr_list.extend([path_event_chain_label, path_event_chain_label])

    # Context snapshot edges
    for edge in snap_edges:
        s_id = str(edge.get("source", ""))
        d_id = str(edge.get("target", ""))
        if s_id in node_id_to_idx and d_id in node_id_to_idx:
            s = node_id_to_idx[s_id]
            d = node_id_to_idx[d_id]
            elabel = edge_vocab.encode(str(edge.get("type", "unknown")))
            src_list.append(s); dst_list.append(d)
            src_list.append(d); dst_list.append(s)
            eattr_list.extend([elabel, elabel])

    if not src_list:
        # No edges — add a self-loop on anchor to avoid empty graph
        anchor_idx = node_id_to_idx.get(anchor_oid, 0)
        src_list = [anchor_idx]; dst_list = [anchor_idx]
        eattr_list = [0]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(eattr_list, dtype=torch.long)

    if _USE_PYG:
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=n_nodes)
    else:
        return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}


# ===========================================================================
# 2.  Model
# ===========================================================================

class HandwrittenGATConv(nn.Module):
    """
    Single-head graph attention layer for use when PyG is absent.
    Implements Equation (1)-(3) of Veličković et al. (2018).
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W   = nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Linear(2 * out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,          # (N, in_dim)
        edge_index: torch.Tensor,  # (2, E)
    ) -> torch.Tensor:             # (N, out_dim)
        h = self.W(x)              # (N, out_dim)
        src, dst = edge_index[0], edge_index[1]
        alpha = torch.cat([h[src], h[dst]], dim=-1)  # (E, 2*out_dim)
        alpha = F.leaky_relu(self.att(alpha), 0.2).squeeze(-1)  # (E,)
        # Softmax per destination node
        alpha = torch.zeros(h.size(0), alpha.size(0), device=x.device) \
                      .scatter_(0, dst.unsqueeze(0), alpha.unsqueeze(0))
        alpha = F.softmax(alpha, dim=-1)  # (N, E) — sparse
        # Aggregate
        agg = torch.zeros_like(h)
        for e_idx in range(edge_index.size(1)):
            s = src[e_idx]; d = dst[e_idx]
            agg[d] += alpha[d, e_idx] * h[s]
        return F.elu(agg)


class PathRerankerGNN(nn.Module):
    """
    Graph Attention Network path reranker.

    Architecture
    ------------
    1. Node embedding layer: projects raw one-hot + query_sim features to
       a dense hidden dimension (input_proj).
    2. Edge-type embedding: a learned embedding for each edge type, added
       to source node features before message passing (edge-conditioned GAT
       variant following Schlichtkrull et al., 2018).
    3. Two GAT layers with skip connections.
    4. Global mean pooling → graph-level embedding.
    5. Two-layer MLP → scalar score.

    The scalar score for a (query, path) pair is then used in a pairwise
    margin ranking loss during training and a softmax ranking at inference.
    """

    def __init__(
        self,
        feat_dim:    int,
        hidden_dim:  int = 128,
        n_edge_types: int = 16,
        n_heads:     int = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.feat_dim   = feat_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.edge_emb = nn.Embedding(n_edge_types + 1, hidden_dim,
                                     padding_idx=0)

        if _USE_PYG:
            self.gat1 = GATConv(hidden_dim, hidden_dim // n_heads,
                                 heads=n_heads, dropout=dropout,
                                 edge_dim=hidden_dim)
            self.gat2 = GATConv(hidden_dim, hidden_dim,
                                 heads=1, dropout=dropout,
                                 edge_dim=hidden_dim)
        else:
            self.gat1 = HandwrittenGATConv(hidden_dim, hidden_dim, dropout)
            self.gat2 = HandwrittenGATConv(hidden_dim, hidden_dim, dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _pool(self, x: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
        """Global mean pooling, handling the single-graph (no batch) case."""
        if _USE_PYG and batch is not None:
            return global_mean_pool(x, batch)
        return x.mean(dim=0, keepdim=True)

    def forward_single(self, graph) -> torch.Tensor:
        """
        Score a single path subgraph.

        Parameters
        ----------
        graph : PyG Data or dict with x / edge_index / edge_attr

        Returns
        -------
        Scalar score tensor (shape (1,)).
        """
        if _USE_PYG:
            x, edge_index, edge_attr = (
                graph.x, graph.edge_index, graph.edge_attr
            )
        else:
            x, edge_index, edge_attr = (
                graph["x"], graph["edge_index"], graph["edge_attr"]
            )

        h = self.input_proj(x)

        # Edge features: sum of source-node hidden + edge-type embedding
        e_emb = self.edge_emb(edge_attr.clamp(max=self.edge_emb.num_embeddings - 1))

        if _USE_PYG:
            h1 = self.gat1(h, edge_index, edge_attr=e_emb)
        else:
            h1 = self.gat1(h, edge_index)

        h1 = self.norm1(h + h1)  # skip connection

        if _USE_PYG:
            h2 = self.gat2(h1, edge_index, edge_attr=e_emb)
        else:
            h2 = self.gat2(h1, edge_index)

        h2 = self.norm2(h1 + h2)  # skip connection

        g = self._pool(h2, None)      # (1, hidden_dim)
        return self.score_head(g)     # (1, 1)

    def forward_batch(self, graphs: List) -> torch.Tensor:
        """
        Score a batch of path subgraphs.

        Parameters
        ----------
        graphs : list of PyG Data objects or dicts

        Returns
        -------
        scores : (B,) tensor of raw scores.
        """
        if _USE_PYG:
            batch = Batch.from_data_list(graphs)
            x, edge_index, edge_attr, batch_vec = (
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            h = self.input_proj(x)
            e_emb = self.edge_emb(
                edge_attr.clamp(max=self.edge_emb.num_embeddings - 1)
            )
            h1 = self.gat1(h, edge_index, edge_attr=e_emb)
            h1 = self.norm1(h + h1)
            h2 = self.gat2(h1, edge_index, edge_attr=e_emb)
            h2 = self.norm2(h1 + h2)
            g  = global_mean_pool(h2, batch_vec)  # (B, hidden)
            return self.score_head(g).squeeze(-1)  # (B,)
        else:
            # Fallback: score individually and cat
            scores = [self.forward_single(g) for g in graphs]
            return torch.cat(scores, dim=0).squeeze(-1)


# ===========================================================================
# 3.  Dataset
# ===========================================================================

class PathRerankingDataset:
    """
    Converts the labelled training instances into (positive, negative) path
    pairs for pairwise margin ranking training.

    Label construction
    ------------------
    For each training instance we have up to k beam paths.  A beam is
    labelled *positive* (label=1) if its activity sequence contains the
    gold_answer token after normalisation, and *negative* (label=0)
    otherwise.  When both positives and negatives exist for the same
    instance we create all (pos, neg) pairs, following the standard
    listwise-to-pairwise reduction (Burges et al., 2005).

    Parameters
    ----------
    instances        : list of training instances (from train_dataset.json)
    predicted_paths  : dict mapping instance_id → list of beam path strings
                       (output of generate_predicted_paths.py, constrained run)
    act_vocab        : fitted VocabEncoder for activities
    obj_vocab        : fitted VocabEncoder for object types
    edge_vocab       : fitted VocabEncoder for edge labels
    """

    def __init__(
        self,
        instances:       List[Dict],
        predicted_paths: Dict[str, List[str]],
        act_vocab:       VocabEncoder,
        obj_vocab:       VocabEncoder,
        edge_vocab:      VocabEncoder,
    ):
        self.instances       = instances
        self.predicted_paths = predicted_paths
        self.act_vocab       = act_vocab
        self.obj_vocab       = obj_vocab
        self.edge_vocab      = edge_vocab

        # Pre-build a simple bag-of-words query embedding (one dimension
        # per activity token) so that build_path_subgraph can receive a
        # query_embedding.  A proper system would use a frozen sentence
        # transformer here (e.g. all-MiniLM-L6-v2, Wang et al. 2020).
        self._query_cache: Dict[str, torch.Tensor] = {}

    def _query_bow(self, question: str) -> torch.Tensor:
        if question in self._query_cache:
            return self._query_cache[question]
        vec = torch.zeros(len(self.act_vocab))
        for tok in question.lower().split():
            idx = self.act_vocab.encode(tok)
            if idx < len(vec):
                vec[idx] += 1.0
        norm = vec.norm()
        if norm > 0:
            vec = vec / norm
        self._query_cache[question] = vec
        return vec

    def _is_positive(self, path_activities: List[str], gold_answer: str) -> bool:
        gold_norm = _normalise_activity(gold_answer)
        combined  = " ".join(path_activities)
        return gold_norm in combined

    def get_pairs(self) -> List[Tuple[Dict, Dict, Dict]]:
        """
        Return a list of (instance_meta, positive_graph, negative_graph)
        triples.  Graphs are not yet converted to tensors (lazy for memory).
        """
        pairs = []
        for inst in self.instances:
            iid         = inst["instance_id"]
            gold_answer = inst.get("gold_answer", "")
            anchor_oid  = inst["anchor_object"]["oid"]
            question    = inst["question"]
            ctx_snap    = inst.get("context_snapshot", {"nodes": [], "edges": []})
            beams       = self.predicted_paths.get(iid, [])

            if not beams:
                continue

            pos_beams, neg_beams = [], []
            for b in beams:
                acts = path_string_to_activities(b)
                if acts and self._is_positive(acts, gold_answer):
                    pos_beams.append(acts)
                elif acts:
                    neg_beams.append(acts)

            if not pos_beams or not neg_beams:
                continue  # need at least one of each for pairwise training

            q_emb = self._query_bow(question)

            for pos_acts in pos_beams:
                for neg_acts in neg_beams:
                    meta = {
                        "instance_id": iid,
                        "anchor_oid":  anchor_oid,
                        "question":    question,
                        "ctx_snap":    ctx_snap,
                        "q_emb":       q_emb,
                        "pos_acts":    pos_acts,
                        "neg_acts":    neg_acts,
                    }
                    pairs.append(meta)

        return pairs

    def build_graphs(self, meta: Dict) -> Tuple[Optional[object], Optional[object]]:
        """Build the positive and negative subgraphs for one pair."""
        kwargs = dict(
            anchor_oid=meta["anchor_oid"],
            context_snapshot=meta["ctx_snap"],
            act_vocab=self.act_vocab,
            obj_vocab=self.obj_vocab,
            edge_vocab=self.edge_vocab,
            query_embedding=meta["q_emb"],
        )
        pos_graph = build_path_subgraph(meta["pos_acts"], **kwargs)
        neg_graph = build_path_subgraph(meta["neg_acts"], **kwargs)
        return pos_graph, neg_graph


# ===========================================================================
# 4.  Vocabulary fitting
# ===========================================================================

def fit_vocabularies(
    instances: List[Dict],
    predicted_paths: Dict[str, List[str]],
) -> Tuple[VocabEncoder, VocabEncoder, VocabEncoder]:
    """
    Build activity, object-type and edge-label vocabularies from the
    training corpus.  Vocabularies must be fitted before any graph
    featurisation so that feature dimensions are consistent across all
    instances.
    """
    act_vocab  = VocabEncoder()
    obj_vocab  = VocabEncoder()
    edge_vocab = VocabEncoder()

    all_activities, all_obj_types, all_edge_labels = [], [], []

    for inst in instances:
        # Object types from context snapshot
        for node in inst.get("context_snapshot", {}).get("nodes", []):
            obj_type = node.get("ocel:type", node.get("type", "unknown"))
            all_obj_types.append(str(obj_type))
        # Edge labels
        for edge in inst.get("context_snapshot", {}).get("edges", []):
            all_edge_labels.append(str(edge.get("type", "unknown")))
        # Gold paths as activity vocabulary seed
        for path_set in inst.get("gold_paths", []):
            for seq in (path_set if isinstance(path_set[0], list) else [path_set]):
                for act in seq:
                    all_activities.append(_normalise_activity(str(act)))

    # Activities from predicted beams (may contain novel activities not in gold)
    for beams in predicted_paths.values():
        for b in beams:
            all_activities.extend(path_string_to_activities(b))

    # Internal edge labels
    all_edge_labels.extend(["path_sequence", "unknown"])

    act_vocab.fit(all_activities)
    obj_vocab.fit(all_obj_types)
    edge_vocab.fit(all_edge_labels)

    print(
        f"Vocabularies: {len(act_vocab)} activities, "
        f"{len(obj_vocab)} object types, "
        f"{len(edge_vocab)} edge labels."
    )
    return act_vocab, obj_vocab, edge_vocab


# ===========================================================================
# 5.  Training loop
# ===========================================================================

def train(
    train_instances:  List[Dict],
    predicted_paths:  Dict[str, List[str]],
    *,
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
    Train the GNN reranker on the training split.

    Parameters
    ----------
    train_instances  : Instances from train_dataset.json (anchor objects
                       fully disjoint from sampled_100, per the split
                       produced by generate_eval_dataset.py).
    predicted_paths  : Dict mapping instance_id → list[str] beam paths,
                       produced by generate_predicted_paths.py (constrained
                       run) over the training instances.
    hidden_dim       : GNN hidden dimension.
    n_heads          : Number of attention heads in GAT layers.
    dropout          : Dropout probability.
    margin           : Hinge margin for the pairwise ranking loss.
    lr               : Learning rate (AdamW).
    weight_decay     : L2 regularisation strength.
    n_epochs         : Number of training epochs.
    batch_size       : Number of (pos, neg) pairs per gradient step.
    seed             : Random seed for reproducibility.
    device_str       : Torch device string ("cpu", "cuda", "mps").
    checkpoint_path  : File path for the best model checkpoint.

    Returns
    -------
    Trained PathRerankerGNN model in eval mode.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_str)

    # ------------------------------------------------------------------ #
    # Fit vocabularies and build dataset
    # ------------------------------------------------------------------ #
    act_vocab, obj_vocab, edge_vocab = fit_vocabularies(
        train_instances, predicted_paths
    )

    dataset = PathRerankingDataset(
        train_instances, predicted_paths, act_vocab, obj_vocab, edge_vocab
    )
    pairs = dataset.get_pairs()
    print(f"Training pairs (pos, neg): {len(pairs)}")

    if len(pairs) == 0:
        raise ValueError(
            "No training pairs found.  Check that predicted_paths contains "
            "beam outputs for at least some training instances, and that some "
            "beams are labelled positive (contain gold_answer) while others "
            "are labelled negative."
        )

    random.shuffle(pairs)

    # ------------------------------------------------------------------ #
    # Determine feature dimension (must be consistent across all graphs)
    # ------------------------------------------------------------------ #
    feat_dim = len(act_vocab) + len(obj_vocab) + 2
    n_edge_types = len(edge_vocab)

    model = PathRerankerGNN(
        feat_dim=feat_dim,
        hidden_dim=hidden_dim,
        n_edge_types=n_edge_types,
        n_heads=n_heads,
        dropout=dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    best_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch_start in tqdm(
            range(0, len(pairs), batch_size),
            desc=f"Epoch {epoch}/{n_epochs}",
            leave=False,
        ):
            batch_meta = pairs[batch_start : batch_start + batch_size]

            pos_graphs, neg_graphs = [], []
            valid_batch = []

            for meta in batch_meta:
                pg, ng = dataset.build_graphs(meta)
                if pg is None or ng is None:
                    continue
                # Move tensors to device
                if _USE_PYG:
                    pg = pg.to(device)
                    ng = ng.to(device)
                else:
                    pg = {k: v.to(device) for k, v in pg.items()}
                    ng = {k: v.to(device) for k, v in ng.items()}
                pos_graphs.append(pg)
                neg_graphs.append(ng)
                valid_batch.append(meta)

            if not pos_graphs:
                continue

            pos_scores = model.forward_batch(pos_graphs)  # (B,)
            neg_scores = model.forward_batch(neg_graphs)  # (B,)

            # Pairwise margin ranking loss
            # loss_i = max(0, margin - pos_score_i + neg_score_i)
            loss = F.relu(
                margin - pos_scores + neg_scores
            ).mean()

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
            checkpoint = {
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "act_vocab":    act_vocab._tok2idx,
                "obj_vocab":    obj_vocab._tok2idx,
                "edge_vocab":   edge_vocab._tok2idx,
                "feat_dim":     feat_dim,
                "hidden_dim":   hidden_dim,
                "n_edge_types": n_edge_types,
                "n_heads":      n_heads,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Checkpoint saved (loss {best_loss:.4f})")

    model.eval()
    return model


# ===========================================================================
# 6.  Inference / re-ranking
# ===========================================================================

def load_reranker(checkpoint_path: str, device_str: str = "cpu") -> Tuple[
    PathRerankerGNN, VocabEncoder, VocabEncoder, VocabEncoder
]:
    """Load a trained reranker from a checkpoint file."""
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


def rerank_paths(
    model:          PathRerankerGNN,
    beams:          List[str],
    anchor_oid:     str,
    question:       str,
    context_snapshot: Dict,
    act_vocab:      VocabEncoder,
    obj_vocab:      VocabEncoder,
    edge_vocab:     VocabEncoder,
    device_str:     str = "cpu",
) -> List[Tuple[str, float]]:
    """
    Re-rank a list of beam paths using the trained GNN reranker.

    Parameters
    ----------
    model           : Trained PathRerankerGNN (eval mode).
    beams           : List of raw beam path strings (from GCR decoding).
    anchor_oid      : Anchor object identifier.
    question        : Natural-language query string.
    context_snapshot: Instance context snapshot dict.
    act_vocab       : Fitted activity vocabulary encoder.
    obj_vocab       : Fitted object-type vocabulary encoder.
    edge_vocab      : Fitted edge-label vocabulary encoder.
    device_str      : Torch device string.

    Returns
    -------
    List of (path_string, score) tuples, sorted by score descending.
    """
    device = torch.device(device_str)
    model.eval()

    # Build query embedding (same BoW as during training)
    q_vec = torch.zeros(len(act_vocab))
    for tok in question.lower().split():
        idx = act_vocab.encode(tok)
        if idx < len(q_vec):
            q_vec[idx] += 1.0
    norm = q_vec.norm()
    if norm > 0:
        q_vec = q_vec / norm

    graphs, valid_beams = [], []
    for b in beams:
        acts = path_string_to_activities(b)
        if not acts:
            continue
        g = build_path_subgraph(
            acts, anchor_oid, context_snapshot,
            act_vocab, obj_vocab, edge_vocab, q_vec
        )
        if g is not None:
            graphs.append(g.to(device) if _USE_PYG else
                          {k: v.to(device) for k, v in g.items()})
            valid_beams.append(b)

    if not graphs:
        return [(b, 0.0) for b in beams]

    with torch.no_grad():
        scores = model.forward_batch(graphs).cpu().tolist()

    ranked = sorted(zip(valid_beams, scores), key=lambda x: x[1], reverse=True)

    # Append any beams that were skipped (no activities) at the bottom
    ranked_set = {b for b, _ in ranked}
    for b in beams:
        if b not in ranked_set:
            ranked.append((b, float("-inf")))

    return ranked


# ===========================================================================
# 7.  CLI
# ===========================================================================

def _load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _load_json(path: str) -> object:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_predicted_paths(jsonl_path: str) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for rec in _load_jsonl(jsonl_path):
        iid   = rec.get("instance_id", "")
        beams = rec.get("paths", [])
        result[iid] = [b for b in beams if b]
    return result


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Train (or run inference with) the GNN path reranker."
    )
    p.add_argument("--mode", choices=["train", "rerank"], default="train")
    # Inputs
    p.add_argument("--train_dataset",   required=False,
                   help="train_dataset.json from generate_eval_dataset.py")
    p.add_argument("--train_paths",     required=False,
                   help="predicted_paths_constrained.jsonl over training set")
    p.add_argument("--eval_dataset",    required=False,
                   help="sampled_100.json (for inference demo)")
    p.add_argument("--eval_paths",      required=False,
                   help="predicted_paths_constrained.jsonl over eval set")
    # Model / training
    p.add_argument("--checkpoint",      default="gnn_reranker.pt")
    p.add_argument("--hidden_dim",      type=int,   default=128)
    p.add_argument("--n_heads",         type=int,   default=4)
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--margin",          type=float, default=0.5)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--n_epochs",        type=int,   default=20)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--device",          default="cpu")
    args = p.parse_args()

    if args.mode == "train":
        assert args.train_dataset and args.train_paths, (
            "--train_dataset and --train_paths are required for training."
        )
        content = open(args.train_dataset, encoding="utf-8").read().strip()
        train_instances = (json.loads(content) if content.startswith("[")
                           else [json.loads(l) for l in content.splitlines() if l])
        predicted_paths = _load_predicted_paths(args.train_paths)

        train(
            train_instances, predicted_paths,
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
        assert args.eval_dataset and args.eval_paths and args.checkpoint, (
            "--eval_dataset, --eval_paths, and --checkpoint are required for reranking."
        )
        model, act_vocab, obj_vocab, edge_vocab = load_reranker(
            args.checkpoint, device_str=args.device
        )
        content = open(args.eval_dataset, encoding="utf-8").read().strip()
        eval_instances = (json.loads(content) if content.startswith("[")
                          else [json.loads(l) for l in content.splitlines() if l])
        eval_paths = _load_predicted_paths(args.eval_paths)

        for inst in eval_instances[:5]:  # demo: first 5
            iid    = inst["instance_id"]
            beams  = eval_paths.get(iid, [])
            ranked = rerank_paths(
                model, beams,
                anchor_oid=inst["anchor_object"]["oid"],
                question=inst["question"],
                context_snapshot=inst.get("context_snapshot",
                                          {"nodes": [], "edges": []}),
                act_vocab=act_vocab, obj_vocab=obj_vocab, edge_vocab=edge_vocab,
                device_str=args.device,
            )
            print(f"\n[{iid}] {inst['question']}")
            print(f"  Gold: {inst['gold_answer']}")
            for rank, (path, score) in enumerate(ranked, 1):
                print(f"  #{rank} (score={score:.3f}): {path[:80]}")