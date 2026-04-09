import time
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

from .trie import ProcessTrie
from .gcr2 import enumerate_object_valid_paths, linearize_event_path

class GCRProcessProcessor(torch.nn.Module):
    """
    Trie-based constrained decoding for GCR path generation.
    """

    def __init__(self, trie: ProcessTrie, prompt_lens: List[int], tokenizer):
        super().__init__()
        self.trie = trie
        self.prompt_lens = prompt_lens
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, input_ids, scores):
        batch_size = input_ids.shape[0]
        num_prompts = len(self.prompt_lens)
        beam_width = max(1, batch_size // num_prompts)

        for b in range(batch_size):
            prompt_idx = min(b // beam_width, num_prompts - 1)
            p_len = self.prompt_lens[prompt_idx]

            generated = input_ids[b][p_len:].tolist()
            allowed = self.trie.allowed_next(generated)

            mask = torch.full_like(scores[b], float("-inf"))

            if allowed:
                allowed = [
                    t for t in allowed
                    if 0 <= t < scores.shape[-1]
                ]
                mask[allowed] = scores[b, allowed]
            else:
                mask[self.eos_token_id] = scores[b, self.eos_token_id]

            scores[b] = mask

        return scores
    
class GCRProcessAgent:
    """
    Single-graph GCR agent using event-centric object-valid paths.
    """

    def __init__(
        self,
        model_id: str,
        events: Dict[str, object],
        event_successors: Dict[str, List[object]],
        device: str = "cpu",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=device,
        )
        self.device = device
        self.events = events
        self.event_successors = event_successors

    def _build_trie(
            self,
            start_events: List[object],
            anchor_object: str,
            max_depth: int,
        ) -> ProcessTrie:

            paths = enumerate_object_valid_paths(
                event_successors=self.event_successors,
                start_events=start_events,
                anchor_object=anchor_object,
                max_depth=max_depth,
            )

            trie = ProcessTrie()

            for path in paths:
                text = linearize_event_path(path)
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                if ids:
                    trie.insert(ids)

            return trie
    
    def generate_paths(
        self,
        start_events: List[object],
        anchor_object: str,
        question: str,
        num_paths: int = 3,
        max_depth: int = 5,
        max_new_tokens: int = 100,
    ) -> List[str]:

        prompt = (f"Question: {question}\n"
                 f"Context: Found object {anchor_object}.\n"
                    f"Reasoning Path: ")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        trie = self._build_trie(start_events, anchor_object, max_depth)

        processor = LogitsProcessorList([
            GCRProcessProcessor(trie, [prompt_len], self.tokenizer)
        ])

        outputs = self.model.generate(
            **inputs,
            logits_processor=processor,
            num_beams=num_paths,
            num_return_sequences=num_paths,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
        )

        return [
            self.tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
            for o in outputs
        ]
    
    def generate_unconstrained(
        self,
        question: str,
        num_paths: int = 3,
        max_new_tokens: int = 1,
    ) -> List[str]:

        prompt = f"Question: {question}\nReasoning Path: "      #TODO add context
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        outputs = self.model.generate(
            **inputs,
            num_beams=num_paths,
            num_return_sequences=num_paths,
            max_new_tokens=max_new_tokens,
        )

        return [
            self.tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
            for o in outputs
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

