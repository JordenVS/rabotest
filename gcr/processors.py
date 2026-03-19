"""
gcr/processors.py
-----------------
GCR logits processor and high-level agent for OCEL 2.0 process graphs.

"""

import time
import torch
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from typing import List

# Internal imports — only functions that actually exist in gcr.gcr
from .gcr import collect_unique_path_strings, build_trie_from_path_strings


# ---------------------------------------------------------------------------
# 1.  CONSTRAINED LOGITS PROCESSOR  (GCR core)
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

    def __init__(self, trie, prompt_lens: List[int], tokenizer):
        super().__init__()
        self.trie = trie
        self.prompt_lens = prompt_lens
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:

        batch_size = input_ids.shape[0]
        num_prompts = len(self.prompt_lens)

        # beam_width: how many beams per original prompt
        # Guard: if batch_size < num_prompts something is very wrong,
        #        but at least avoid ZeroDivisionError.
        beam_width = max(1, batch_size // num_prompts)

        for b in range(batch_size):
            prompt_idx = min(b // beam_width, num_prompts - 1)
            p_len = self.prompt_lens[prompt_idx]

            # Tokens generated so far for this beam (excluding the prompt)
            current_gen_ids = input_ids[b][p_len:].tolist()
            allowed = self.trie.allowed_next(current_gen_ids)

            mask = torch.full_like(scores[b], float("-inf"))

            if allowed:
                allowed_tensor = torch.tensor(
                    list(allowed), dtype=torch.long, device=scores.device
                )
                # Clamp to actual vocab size (safety guard)
                vocab_size = scores.shape[-1]
                allowed_tensor = allowed_tensor[
                    (allowed_tensor >= 0) & (allowed_tensor < vocab_size)
                ]
                mask[allowed_tensor] = scores[b, allowed_tensor]
            else:
                # Trie exhausted — allow EOS so generation terminates cleanly
                mask[self.eos_token_id] = scores[b, self.eos_token_id]

            scores[b] = mask

        return scores


# ---------------------------------------------------------------------------
# 2.  HIGH-LEVEL AGENT
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_trie_for_seed(self, seed_node: str, max_depth: int):
        """Build a ProcessTrie from all paths starting at *seed_node*."""
        path_strings = collect_unique_path_strings(
            self.graph, [seed_node], max_depth=max_depth
        )
        return build_trie_from_path_strings(path_strings, self.tokenizer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_compliant_paths(
        self,
        seed_entity: str,
        question: str,
        num_paths: int = 3,
        max_depth: int = 4,
        max_new_tokens: int = 100,
    ) -> List[str]:
        """
        Generate *num_paths* graph-constrained reasoning paths starting from
        *seed_entity* in response to *question*.

        Returns
        -------
        List[str]
            Decoded path strings (one per beam), guaranteed to be valid walks
            in self.graph.
        """
        # A. Prompt
        prompt = (
            f"Question: {question}\n"
            f"Context: Found Object {seed_entity}.\n"
            f"Reasoning Path: "
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        # B. Build trie for this seed
        process_trie = self._build_trie_for_seed(seed_entity, max_depth=max_depth)

        # C. Constrained logits processor
        logits_processor = LogitsProcessorList(
            [GCRProcessProcessor(process_trie, [prompt_len], self.tokenizer)]
        )

        # D. Constrained beam search
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_paths,
            num_return_sequences=num_paths,
            logits_processor=logits_processor,
            early_stopping=True,
        )

        # E. Decode (strip the prompt prefix)
        paths = [
            self.tokenizer.decode(g[prompt_len:], skip_special_tokens=True)
            for g in output_ids
        ]
        return paths

    def generate_unconstrained_paths(
        self,
        seed_entity: str,
        question: str,
        num_paths: int = 3,
        max_new_tokens: int = 100,
    ) -> List[str]:
        """
        Identical prompt and beam search, but *without* trie constraints.
        Used as the baseline ('GCR w/o constraint') in the ablation study,
        directly mirroring Figure 5 of Luo et al. (2024).
        """
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

        paths = [
            self.tokenizer.decode(g[prompt_len:], skip_special_tokens=True)
            for g in output_ids
        ]
        return paths

    def timed_generate(
        self,
        seed_entity: str,
        question: str,
        constrained: bool = True,
        num_paths: int = 3,
        max_depth: int = 4,
    ) -> dict:
        """
        Wrapper that records wall-clock timing for the efficiency analysis
        table (cf. Table 2 in Luo et al., 2024).

        Returns
        -------
        dict with keys: paths, trie_build_s, generation_s, prompt_tokens
        """
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
