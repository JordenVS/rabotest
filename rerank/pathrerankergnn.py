from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    _USE_PYG = True
except ImportError:
    _USE_PYG = False


class _HandwrittenGATConv(nn.Module):
    """
    Single-head GAT layer (Veličković et al., 2018) for use when PyG is
    absent.  Implements attention score → per-destination softmax → aggregate.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W    = nn.Linear(in_dim,      out_dim, bias=False)
        self.att  = nn.Linear(2 * out_dim, 1,       bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h        = self.W(x)
        src, dst = edge_index[0], edge_index[1]
        e_score  = F.leaky_relu(
            self.att(torch.cat([h[src], h[dst]], dim=-1)), 0.2
        ).squeeze(-1)
        # Per-destination softmax (sparse, via scatter)
        alpha = torch.full((h.size(0), src.size(0)), float("-inf"),
                           device=x.device)
        for ei in range(src.size(0)):
            alpha[dst[ei], ei] = e_score[ei]
        alpha = F.softmax(alpha, dim=-1)   # (N, E)
        agg = torch.zeros_like(h)
        for ei in range(src.size(0)):
            agg[dst[ei]] += alpha[dst[ei], ei] * h[src[ei]]
        return F.elu(agg)


class PathRerankerGNN(nn.Module):
    """
    Two-layer GAT reranker with residual connections and global mean pooling.

    Input  : a batch of path subgraphs (PyG Data list or dict list)
    Output : (B,) tensor of raw relevance scores — higher = better path
    """

    def __init__(
        self,
        feat_dim:     int,
        hidden_dim:   int   = 128,
        n_edge_types: int   = 16,
        n_heads:      int   = 4,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.edge_emb = nn.Embedding(n_edge_types + 1, hidden_dim, padding_idx=0)

        if _USE_PYG:
            self.gat1 = GATConv(hidden_dim, hidden_dim // n_heads,
                                 heads=n_heads, dropout=dropout, edge_dim=hidden_dim)
            self.gat2 = GATConv(hidden_dim, hidden_dim,
                                 heads=1,       dropout=dropout, edge_dim=hidden_dim)
        else:
            self.gat1 = _HandwrittenGATConv(hidden_dim, hidden_dim, dropout)
            self.gat2 = _HandwrittenGATConv(hidden_dim, hidden_dim, dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _encode_graph(self, x, edge_index, edge_attr, batch=None):
        h  = self.input_proj(x)
        ee = self.edge_emb(edge_attr.clamp(max=self.edge_emb.num_embeddings - 1))
        if _USE_PYG:
            h1 = self.norm1(h  + self.gat1(h,  edge_index, edge_attr=ee))
            h2 = self.norm2(h1 + self.gat2(h1, edge_index, edge_attr=ee))
            return global_mean_pool(h2, batch)          # (B, hidden)
        else:
            h1 = self.norm1(h  + self.gat1(h,  edge_index))
            h2 = self.norm2(h1 + self.gat2(h1, edge_index))
            return h2.mean(dim=0, keepdim=True)         # (1, hidden)

    def forward_batch(self, graphs: List) -> torch.Tensor:
        """Score a list of subgraphs; returns (B,) tensor."""
        if _USE_PYG:
            b = Batch.from_data_list(graphs)
            g = self._encode_graph(b.x, b.edge_index, b.edge_attr, b.batch)
        else:
            g = torch.cat([
                self._encode_graph(gr["x"], gr["edge_index"], gr["edge_attr"])
                for gr in graphs
            ], dim=0)
        return self.score_head(g).squeeze(-1)            # (B,)

