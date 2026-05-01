from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project imports — Event objects and graph builders
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gcr.event import Event
from rerank.train import build_path_subgraph, VocabEncoder

class PathRerankingDataset:
    """
    Generates (positive, negative) path pairs on-the-fly via object-valid
    random walks over G_behavior — no pre-computed beam files required.

    For each training instance *num_paths* walks are generated from the
    anchor object using generate_walks().  Walks containing the gold_answer
    activity (normalised substring match) are labelled positive; all others
    negative.  Every (pos, neg) combination is emitted as a training pair,
    following the listwise-to-pairwise reduction of Burges et al. (2005).
    """

    def __init__(
        self,
        instances:        List[Dict],
        events:           Dict[str, Event],
        event_successors: Dict[str, List[Event]],
        act_vocab:        VocabEncoder,
        obj_vocab:        VocabEncoder,
        edge_vocab:       VocabEncoder,
        num_paths:        int   = 5,
        max_depth:        int   = 7,
        seed:             int   = 42,
    ):
        self.instances        = instances
        self.events           = events
        self.event_successors = event_successors
        self.act_vocab        = act_vocab
        self.obj_vocab        = obj_vocab
        self.edge_vocab       = edge_vocab
        self.num_paths        = num_paths
        self.max_depth        = max_depth
        self.seed             = seed
        self._q_cache: Dict[str, torch.Tensor] = {}

    def _query_bow(self, question: str) -> torch.Tensor:
        if question in self._q_cache:
            return self._q_cache[question]
        vec = torch.zeros(len(self.act_vocab))
        for tok in question.lower().split():
            idx = self.act_vocab.encode(tok)
            if idx < len(vec):
                vec[idx] += 1.0
        norm = vec.norm()
        if norm > 0:
            vec = vec / norm
        self._q_cache[question] = vec
        return vec

    @staticmethod
    def _is_positive(path_activities: List[str], gold_answer: str) -> bool:
        return _normalise_activity(gold_answer) in " ".join(path_activities)

    def get_pairs(self) -> List[Dict]:
        """
        Materialise all training pairs as plain dicts with keys:
        anchor_oid, question, ctx_snap, q_emb, pos_acts, neg_acts.
        """
        pairs: List[Dict] = []

        for inst_idx, inst in enumerate(self.instances):
            anchor_oid  = inst["anchor_object"]["oid"]
            gold_answer = inst.get("gold_answer", "")
            question    = inst["question"]
            ctx_snap    = inst.get("context_snapshot", {"nodes": [], "edges": []})
            rng         = random.Random(self.seed + inst_idx)

            walk_strings = generate_walks(
                anchor_oid, self.events, self.event_successors,
                self.num_paths, self.max_depth, rng,
            )

            pos_list: List[List[str]] = []
            neg_list: List[List[str]] = []
            for ws in walk_strings:
                acts = _activities_from_path_string(ws)
                if not acts:
                    continue
                if self._is_positive(acts, gold_answer):
                    pos_list.append(acts)
                else:
                    neg_list.append(acts)

            if not pos_list or not neg_list:
                continue   # need at least one of each for pairwise loss

            q_emb = self._query_bow(question)
            for pos_acts in pos_list:
                for neg_acts in neg_list:
                    pairs.append({
                        "anchor_oid": anchor_oid,
                        "question":   question,
                        "ctx_snap":   ctx_snap,
                        "q_emb":      q_emb,
                        "pos_acts":   pos_acts,
                        "neg_acts":   neg_acts,
                    })

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
        return (
            build_path_subgraph(meta["pos_acts"], **kwargs),
            build_path_subgraph(meta["neg_acts"], **kwargs),
        )
