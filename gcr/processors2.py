"""
gcr/processors.py
-----------------
GCR logits processor and high-level agents for OCEL 2.0 process graphs.

Classes
-------
GCRProcessProcessor   — logits processor (shared by both agents)
GCRProcessAgent       — single-graph constrained decoding (original)
DualGCRProcessAgent   — separate local + global graph decoding
"""

import time
import torch
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from typing import List, Tuple, Optional

from .gcr import collect_unique_path_strings, build_trie_from_path_strings


# ---------------------------------------------------------------------------
# 1.  CONSTRAINED LOGITS PROCESSOR  (shared by both agents)
# ---------------------------------------------------------------------------

class GCRProcessProcessor(torch.nn.Module):
    """
    Masks the LLM vocabulary at every decoding step so that only token
    continuations that form valid paths in the OCEL graph's ProcessTrie
    are allowed.  This is the direct adaptation of the KG-Trie constrained
    decoding described in Luo et al. (2024).

    Parameters
    ----------
    trie : ProcessTrie
        Trie built from the graph paths rooted at the seed entity.
    prompt_lens : List[int]
        Length (in tokens) of each prompt in the batch *before* generation
        starts.  Used to strip the prompt prefix when querying the trie.
    tokenizer : PreTrainedTokenizer
        The same tokenizer used to build the trie and encode the prompt.
    """

    def __init__(self, trie, prompt_lens: List[int], tokenizer, edge_boost: float = 3.0):
        super().__init__()
        self.trie = trie
        self.prompt_lens = prompt_lens
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

        self.next_for_token_ids = {
            tid for tok, tid in tokenizer.get_vocab().items()
            if "NEXT_FOR" in tok
        }
        self.edge_boost = edge_boost

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:

        batch_size = input_ids.shape[0]
        num_prompts = len(self.prompt_lens)
        beam_width = max(1, batch_size // num_prompts)

        for b in range(batch_size):
            prompt_idx = min(b // beam_width, num_prompts - 1)
            p_len = self.prompt_lens[prompt_idx]

            current_gen_ids = input_ids[b][p_len:].tolist()
            allowed = self.trie.allowed_next(current_gen_ids)

            mask = torch.full_like(scores[b], float("-inf"))

            if allowed:
                allowed_tensor = torch.tensor(
                    list(allowed), dtype=torch.long, device=scores.device
                )
                vocab_size = scores.shape[-1]
                allowed_tensor = allowed_tensor[
                    (allowed_tensor >= 0) & (allowed_tensor < vocab_size)
                ]
                mask[allowed_tensor] = scores[b, allowed_tensor]
                # 1. copy original scores for all allowed tokens
                mask[allowed_tensor] = scores[b, allowed_tensor]

                # 2. boost NEXT_FOR tokens on top of their original scores
                if self.edge_boost != 0.0:
                    boost_candidates = allowed_tensor[
                        torch.isin(
                            allowed_tensor,
                            torch.tensor(
                                list(self.next_for_token_ids),
                                dtype=torch.long,
                                device=scores.device,
                            )
                        )
                    ]
                    mask[boost_candidates] += self.edge_boost
            else:
                mask[self.eos_token_id] = scores[b, self.eos_token_id]

            scores[b] = mask


        return scores


# ---------------------------------------------------------------------------
# 2.  ORIGINAL SINGLE-GRAPH AGENT  (unchanged)
# ---------------------------------------------------------------------------

class GCRProcessAgent:
    """
    End-to-end agent that, given a seed entity and a natural-language
    question, returns the top-K process paths constrained to the OCEL graph.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier (e.g. "Qwen/Qwen2.5-1.5B-Instruct").
    graph : nx.DiGraph
        NetworkX graph produced by ocel_to_graph_with_pm4py.
    device : str
        "cpu" or "cuda".
    """

    def __init__(self, model_id: str, graph: nx.DiGraph, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=device,
        )
        self.device = device
        self.graph = graph

    def _build_trie_for_seed(self, seed_node: str, max_depth: int):
        path_strings = collect_unique_path_strings(
            self.graph, [seed_node], max_depth=max_depth
        )
        return build_trie_from_path_strings(path_strings, self.tokenizer)

    def generate_compliant_paths(
        self,
        seed_entity: str,
        question: str,
        num_paths: int = 3,
        max_depth: int = 4,
        max_new_tokens: int = 100,
    ) -> List[str]:
        prompt = (
            f"Question: {question}\n"
            f"Context: Found Object {seed_entity}.\n"
            f"Reasoning Path: "
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        process_trie = self._build_trie_for_seed(seed_entity, max_depth=max_depth)

        logits_processor = LogitsProcessorList(
            [GCRProcessProcessor(process_trie, [prompt_len], self.tokenizer)]
        )

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_paths,
            num_return_sequences=num_paths,
            logits_processor=logits_processor,
            early_stopping=True,
        )

        return [
            self.tokenizer.decode(g[prompt_len:], skip_special_tokens=True)
            for g in output_ids
        ]

    def generate_unconstrained_paths(
        self,
        seed_entity: str,
        question: str,
        num_paths: int = 3,
        max_new_tokens: int = 100,
    ) -> List[str]:
        prompt = (
            f"Question: {question}\n"
            f"Context: Found Object {seed_entity}.\n"
            f"Reasoning Path: "
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_paths,
            num_return_sequences=num_paths,
            early_stopping=True,
        )

        return [
            self.tokenizer.decode(g[prompt_len:], skip_special_tokens=True)
            for g in output_ids
        ]

    def timed_generate(
        self,
        seed_entity: str,
        question: str,
        constrained: bool = True,
        num_paths: int = 3,
        max_depth: int = 4,
    ) -> dict:
        inputs = self.tokenizer(
            f"Question: {question}\nContext: Found Object {seed_entity}.\nReasoning Path: ",
            return_tensors="pt",
        )
        prompt_tokens = inputs.input_ids.shape[1]

        trie_build_s = 0.0
        if constrained:
            t0 = time.perf_counter()
            _ = self._build_trie_for_seed(seed_entity, max_depth=max_depth)
            trie_build_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        if constrained:
            paths = self.generate_compliant_paths(
                seed_entity, question, num_paths=num_paths, max_depth=max_depth
            )
        else:
            paths = self.generate_unconstrained_paths(
                seed_entity, question, num_paths=num_paths
            )
        generation_s = time.perf_counter() - t1

        return {
            "paths": paths,
            "trie_build_s": trie_build_s,
            "generation_s": generation_s,
            "total_s": trie_build_s + generation_s,
            "prompt_tokens": prompt_tokens,
        }


# ---------------------------------------------------------------------------
# 3.  DUAL-GRAPH AGENT  (new)
# ---------------------------------------------------------------------------

class DualGCRProcessAgent:
    """
    Dual-context GCR agent that maintains separate local (instance-level) and
    global (population-level) graphs and generates independently constrained
    path sets from each.

    The two path sets are combined into a structured prompt that presents both
    the specific case behaviour and the typical process behaviour to the LLM,
    allowing it to synthesise a conformance-aware answer.

    This is distinct from the integrated graph approach: no cross-layer trie
    edges exist — each trie constrains only within its own graph layer.
    Cross-layer synthesis is delegated to the LLM at generation time.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    G_local : nx.DiGraph
        Instance-level graph from ocel_to_graph_with_pm4py().
    G_global : nx.DiGraph
        Population-level OC-DFG graph from build_global_context_from_ocel().
        Activity node IDs must follow the "activity:<name>" convention.
    global_max_depth : int
        Max DFG hops for global trie construction.  Defaults to 3 — the
        global graph is typically small enough that depth 3 is sufficient
        and avoids combinatorial path explosion.
    device : str
        "cpu" or "cuda".
    """

    def __init__(
        self,
        model_id: str,
        G_local: nx.DiGraph,
        G_global: nx.DiGraph,
        global_max_depth: int = 3,
        device: str = "cpu",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=device,
        )
        self.device = device
        self.G_local = G_local
        self.G_global = G_global

        # Build the global trie once at init — it does not change per seed.
        # Start nodes are activity nodes that have no DFG predecessors, i.e.
        # the process entry points.  Fall back to all activity nodes if none
        # are identified (e.g. in cyclic graphs).
        global_start_nodes = self._get_global_start_nodes()
        print(f"[DualGCR] Building global trie from "
              f"{len(global_start_nodes)} start activities...")
        global_path_strings = collect_unique_path_strings(
            G_global, global_start_nodes, max_depth=global_max_depth
        )
        self.global_trie = build_trie_from_path_strings(
            global_path_strings, self.tokenizer
        )
        print(f"[DualGCR] Global trie ready "
              f"({len(global_path_strings)} path strings).")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_global_start_nodes(self) -> List[str]:
        """
        Return activity nodes with in-degree 0 in G_global (process entry
        points).  If all nodes have predecessors (cyclic DFG), fall back to
        all activity nodes so the trie still gets built.
        """
        start_nodes = [
            n for n in self.G_global.nodes
            if self.G_global.in_degree(n) == 0
            and self.G_global.nodes[n].get("entity_type") == "Activity"
        ]
        if not start_nodes:
            # Cyclic graph — use all activity nodes as start points
            start_nodes = [
                n for n in self.G_global.nodes
                if self.G_global.nodes[n].get("entity_type") == "Activity"
            ]
        return start_nodes

    def _activity_node_for_seed(self, seed_entity: str) -> Optional[str]:
        """
        Given a local graph seed (event or object node), return the
        corresponding global activity node ID, or None if not found.

        For Event nodes: maps via the 'activity' attribute.
        For Object nodes: not directly mappable — returns None.
        """
        data = self.G_local.nodes.get(seed_entity, {})
        entity_type = data.get("entity_type", "")

        if entity_type == "Event":
            act = data.get("activity", "")
            candidate = f"activity:{act.replace(' ', '_')}"
            return candidate if candidate in self.G_global else None

        # Object node — no direct activity mapping
        return None

    def _generate_constrained(
        self,
        prompt: str,
        trie,
        num_paths: int,
        max_new_tokens: int,
    ) -> List[str]:
        """
        Run beam search with trie-constrained logits processor.
        Returns decoded path strings (prompt prefix stripped).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        logits_processor = LogitsProcessorList(
            [GCRProcessProcessor(trie, [prompt_len], self.tokenizer)]
        )

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_paths,
            num_return_sequences=num_paths,
            logits_processor=logits_processor,
            early_stopping=True,
        )

        return [
            self.tokenizer.decode(g[prompt_len:], skip_special_tokens=True)
            for g in output_ids
        ]

    def _build_local_trie(self, seed_entity: str, max_depth: int):
        path_strings = collect_unique_path_strings(
            self.G_local, [seed_entity], max_depth=max_depth
        )
        return build_trie_from_path_strings(path_strings, self.tokenizer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_dual_paths(
        self,
        seed_entity: str,
        question: str,
        num_paths: int = 3,
        local_max_depth: int = 4,
        max_new_tokens: int = 100,
    ) -> dict:
        """
        Generate independently constrained path sets from both the local
        instance graph and the global OC-DFG graph, then return both together
        with a combined prompt string ready for final LLM generation.

        Parameters
        ----------
        seed_entity : str
            Local graph node ID (e.g. "event:52" or "purchase_order:587").
        question : str
            Natural-language question to answer.
        num_paths : int
            Number of beam paths to generate per layer.
        local_max_depth : int
            Max hops for local trie construction.
        max_new_tokens : int
            Max new tokens per generation call.

        Returns
        -------
        dict with keys:
            local_paths   — List[str] constrained to G_local
            global_paths  — List[str] constrained to G_global (empty list
                            if seed has no corresponding activity node)
            prompt        — str combined prompt for final answer generation
            activity_node — str or None, the global seed node used
        """
        # --- Local generation ---
        local_trie = self._build_local_trie(seed_entity, local_max_depth)
        local_prompt = (
            f"Question: {question}\n"
            f"Context: Found entity {seed_entity}.\n"
            f"Instance reasoning path: "
        )
        local_paths = self._generate_constrained(
            local_prompt, local_trie, num_paths, max_new_tokens
        )

        # --- Global generation ---
        act_node = self._activity_node_for_seed(seed_entity)
        global_paths: List[str] = []

        if act_node is not None:
            global_prompt = (
                f"Question: {question}\n"
                f"Context: Activity {act_node}.\n"
                f"Process reasoning path: "
            )
            global_paths = self._generate_constrained(
                global_prompt, self.global_trie, num_paths, max_new_tokens
            )
        else:
            print(f"[DualGCR] No global activity node found for '{seed_entity}'. "
                  f"Skipping global generation.")

        # --- Build combined prompt ---
        combined_prompt = self._build_combined_prompt(
            question, seed_entity, local_paths, global_paths
        )

        return {
            "local_paths": local_paths,
            "global_paths": global_paths,
            "prompt": combined_prompt,
            "activity_node": act_node,
        }

    def _build_combined_prompt(
        self,
        question: str,
        seed_entity: str,
        local_paths: List[str],
        global_paths: List[str],
    ) -> str:
        """
        Assemble the final LLM prompt combining both path sets.
        The local block grounds the answer in the specific case;
        the global block provides population-level process norms.
        """
        local_block = "\n".join(f"  {p}" for p in local_paths) or "  (none)"
        global_block = "\n".join(f"  {p}" for p in global_paths) or "  (none)"

        return (
            f"You are a process mining assistant for Procure-to-Pay event logs.\n"
            f"Answer the question using the paths below.\n"
            f"Instance paths describe what happened for this specific case.\n"
            f"Process paths describe typical behaviour across all cases.\n\n"
            f"Question: {question}\n"
            f"Seed entity: {seed_entity}\n\n"
            f"Instance-level paths (specific case behaviour):\n{local_block}\n\n"
            f"Population-level paths (typical process behaviour):\n{global_block}\n\n"
            f"Answer:"
        )

    def generate_answer(
        self,
        seed_entity: str,
        question: str,
        num_paths: int = 3,
        local_max_depth: int = 4,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Full pipeline: generate dual path sets, assemble prompt, generate
        final unconstrained answer.

        The final answer generation is unconstrained — the trie constraints
        apply only to path retrieval, not to the answer itself.

        Returns
        -------
        str — the LLM's answer.
        """
        result = self.generate_dual_paths(
            seed_entity, question,
            num_paths=num_paths,
            local_max_depth=local_max_depth,
            max_new_tokens=100,
        )

        inputs = self.tokenizer(
            result["prompt"], return_tensors="pt"
        ).to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        return self.tokenizer.decode(
            output_ids[0][prompt_len:], skip_special_tokens=True
        )

    def timed_generate(
        self,
        seed_entity: str,
        question: str,
        num_paths: int = 3,
        max_depth: int = 4,
    ) -> dict:
        """
        Wrapper that records wall-clock timing broken down by stage, for
        the efficiency analysis table (cf. Table 2 in Luo et al., 2024).

        Returns
        -------
        dict with keys: local_paths, global_paths, prompt, activity_node,
                        local_trie_build_s, local_gen_s, global_gen_s,
                        answer_gen_s, total_s, prompt_tokens
        """
        t_start = time.perf_counter()

        # Local trie build
        t0 = time.perf_counter()
        local_trie = self._build_local_trie(seed_entity, max_depth)
        local_trie_build_s = time.perf_counter() - t0

        # Local generation
        local_prompt = (
            f"Question: {question}\n"
            f"Context: Found entity {seed_entity}.\n"
            f"Instance reasoning path: "
        )
        inputs = self.tokenizer(local_prompt, return_tensors="pt")
        prompt_tokens = inputs.input_ids.shape[1]

        t0 = time.perf_counter()
        local_paths = self._generate_constrained(
            local_prompt, local_trie, num_paths, max_new_tokens=100
        )
        local_gen_s = time.perf_counter() - t0

        # Global generation
        act_node = self._activity_node_for_seed(seed_entity)
        global_paths: List[str] = []
        global_gen_s = 0.0

        if act_node is not None:
            global_prompt = (
                f"Question: {question}\n"
                f"Context: Activity {act_node}.\n"
                f"Process reasoning path: "
            )
            t0 = time.perf_counter()
            global_paths = self._generate_constrained(
                global_prompt, self.global_trie, num_paths, max_new_tokens=100
            )
            global_gen_s = time.perf_counter() - t0

        combined_prompt = self._build_combined_prompt(
            question, seed_entity, local_paths, global_paths
        )

        # Final answer generation
        # inputs = self.tokenizer(combined_prompt, return_tensors="pt").to(self.device)
        # p_len = inputs.input_ids.shape[1]
        # t0 = time.perf_counter()
        # out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        # answer_gen_s = time.perf_counter() - t0
        # answer = self.tokenizer.decode(out[0][p_len:], skip_special_tokens=True)

        return {
            "local_paths": local_paths,
            "global_paths": global_paths,
            "prompt": combined_prompt,
#            "answer": answer,
            "activity_node": act_node,
            "local_trie_build_s": local_trie_build_s,
            "local_gen_s": local_gen_s,
            "global_gen_s": global_gen_s,
#            "answer_gen_s": answer_gen_s,
            "total_s": time.perf_counter() - t_start,
            "prompt_tokens": prompt_tokens,
        }