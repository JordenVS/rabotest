import time
import networkx as nx
import torch
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

from .trie import ProcessTrie
from .gcr import enumerate_object_valid_paths, linearize_event_path, enrich_paths_with_context

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
        trie: Optional[ProcessTrie] = None,
    ) -> List[str]:

        prompt = (f"Question: {question}\n"
                 f"Context: Found object {anchor_object}.\n"
                    f"Reasoning Path: ")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        if trie is None:
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
        anchor_object: str,
        G_context: nx.DiGraph,
        num_paths: int = 3,
        max_new_tokens: int = 100,
        max_hops: int = 2
    ) -> List[str]:
        """
        Enhanced unconstrained generation for ablation studies.
        Provides the LLM with the local graph context in the prompt 
        without enforcing a decoding constraint.
        """
        
        # 1. Extract local neighborhood context from the graph
        # This serves as the "Soft Constraint" (RAG-style)
        subgraph_paths = []
        if anchor_object in G_context:
            # Use simple BFS or path enumeration to get raw graph strings
            # Similar to GCR path extraction but used as prompt text only
            from .gcr import get_raw_neighborhood_strings
            subgraph_paths = get_raw_neighborhood_strings(
                G_context, anchor_object, max_hops=max_hops
            )[:10] # Limit to 10 for prompt efficiency
        
        context_str = "\n".join(subgraph_paths) if subgraph_paths else "No local context found."

        # 2. Build the rich instruction prompt
        prompt = (
            f"You are a process mining assistant. Your task is to generate {num_paths} "
            f"valid process reasoning paths to answer the question starting from the seed entity.\n\n"
            f"### FORMAT RULES:\n"
            f"- Alternate strictly between nodes and relations: Node REL Node REL Node ...\n"
            f"- Event nodes: Event:Activity_Name (use underscores, e.g. Event:Create_Purchase_Order)\n"
            f"- Object nodes: Object:object_type (e.g. Object:goods_receipt)\n"
            f"- Relations: plain lowercase string (e.g. goods_receipt, NEXT_FOR_purchase_order)\n"
            f"- No punctuation, no numbering, no explanation, no newlines\n"
            f"- Output the paths only, nothing else\n\n"
            f"### EXAMPLE PATHS:\n"
            f"Event:Create_Purchase_Order NEXT_FOR_purchase_order Event:Approve_Purchase_Order NEXT_FOR_purchase_order Event:Create_Goods_Receipt\n\n"
            f"### VALID GRAPH CONTEXT FROM SEED:\n"
            f"{context_str}\n\n"
            f"Question: {question}\n"
            f"Seed entity: {anchor_object}\n\n"
            f"Reasoning Path: "
        )

        # 3. Standard model generation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        outputs = self.model.generate(
            **inputs,
            num_beams=num_paths,
            num_return_sequences=num_paths,
            max_new_tokens=max_new_tokens,
        )

        return [
            self.tokenizer.decode(o[prompt_len:], skip_special_tokens=True).strip()
            for o in outputs
        ]
    
    def reify_generated_path(self, generated_string, anchor_object, G_context):
        # 1. Normalize LLM output strings
        activity_names = [
            s.replace("Event:", "").replace("_", " ").strip() 
            for s in generated_string.split()
        ]
        
        # 2. Find candidate EIDs using G_context edges
        # G_context contains "participation" edges between Events and Objects
        if anchor_object not in G_context:
            print(f" [WARNING] Anchor {anchor_object} not in G_context")
            return []

        # Get all event nodes connected to this object in G_context
        candidate_eids = [
            neighbor for neighbor in G_context.neighbors(anchor_object)
            if G_context.nodes[neighbor].get("entity_type") == "Event"
        ]

        # 3. Create Event objects with metadata for sorting
        candidate_events = []
        for eid in candidate_eids:
            node_data = G_context.nodes[eid]
            # We wrap the data back into your Event class or a temporary dict
            candidate_events.append({
                "eid": eid,
                "activity": node_data.get("activity"),
                "timestamp": node_data.get("timestamp", "0000")
            })

        # 4. Sort by timestamp to maintain process-aware order
        candidate_events.sort(key=lambda x: str(x["timestamp"]))

        reified_path = []
        last_idx = -1
        for act_name in activity_names:
            found = False
            for i in range(last_idx + 1, len(candidate_events)):
                if candidate_events[i]["activity"] == act_name:
                    # Retrieve the full Event object from your main storage
                    reified_path.append(self.events[candidate_events[i]["eid"]])
                    last_idx = i
                    found = True
                    break
        
        return reified_path
    
    def timed_generate(
        self,
        anchor_object: str,
        question: str,
        constrained: bool = True,
        enrich: bool = True,
        G_context = None,
        num_paths: int = 3,
        max_depth: int = 5,
        max_new_tokens: int = 100,
    ) -> dict:
        """
        Run constrained or unconstrained path generation and return paths
        together with wall-clock timing and prompt length for efficiency analysis
        (cf. Table 2 in Luo et al., 2025).

        When enrich=True, constrained paths are additionally enriched with
        object context from G_context via enrich_paths_with_context(). This
        adds a context_block key to the return dict — a structured natural-language
        description of the objects and relations on each path, ready for injection
        into the large LLM prompt in run_evaluation.py. G_context must be
        provided when enrich=True.

        Parameters
        ----------
        anchor_object : str
            Query anchor object ID (e.g. "material:835").
        question : str
            Natural-language question.
        constrained : bool
            True  -> trie-constrained beam search (GCR proper).
            False -> unconstrained beam search (ablation baseline).
        enrich : bool
            True -> also run enrich_paths_with_context after decoding.
            Only valid when constrained=True; raises ValueError otherwise.
        G_context : nx.DiGraph | None
            Required when enrich=True.
        num_paths / max_depth / max_new_tokens : int
            Generation hyperparameters.

        Returns
        -------
        dict with keys:
            paths          : List[str]  decoded beam outputs
            trie_build_s   : float      seconds building ProcessTrie (0 if unconstrained)
            generation_s   : float      seconds in model.generate()
            enrich_s       : float      seconds for context enrichment (0 if not enriched)
            total_s        : float      sum of all three
            prompt_tokens  : int        tokens in the shared prompt
            context_block  : str | None structured object context (None if not enriched)
        """
        if enrich and not constrained:
            raise ValueError("enrich=True requires constrained=True.")
        if enrich and G_context is None:
            raise ValueError("enrich=True requires G_context to be provided.")

        start_events = [
            event for event in self.events.values()
            if anchor_object in event.objects
        ]
        if not start_events:
            print(f"  [WARNING] No start events found for object '{anchor_object}'.")

        prompt = (
            f"Question: {question}\n"
            f"Context: Found object {anchor_object}.\n"
            f"Reasoning Path: "
        )
        prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))

        trie_build_s = 0.0
        enrich_s = 0.0
        context_block = None

        if constrained:
            t0 = time.perf_counter()
            trie = self._build_trie(start_events, anchor_object, max_depth)
            trie_build_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            paths = self.generate_paths(
                start_events=start_events,
                anchor_object=anchor_object,
                question=question,
                num_paths=num_paths,
                max_depth=max_depth,
                max_new_tokens=max_new_tokens,
                trie=trie,
            )
            generation_s = time.perf_counter() - t1

        if enrich:
            t2 = time.perf_counter()

            reified_paths = [
                self.reify_generated_path(p, anchor_object, G_context) 
                for p in paths[:num_paths]
            ]
            
            # Now context_block receives Event objects, not strings
            context_block = enrich_paths_with_context(
                paths=reified_paths,
                anchor_object=anchor_object,
                G_context=G_context,
            )
            enrich_s = time.perf_counter() - t2
        else:
            t1 = time.perf_counter()
            paths = self.generate_unconstrained(
                anchor_object=anchor_object,
                question=question,
                num_paths=num_paths,
                max_new_tokens=max_new_tokens,
            )
            generation_s = time.perf_counter() - t1

        return {
            "paths":         paths,
            "trie_build_s":  trie_build_s,
            "generation_s":  generation_s,
            "enrich_s":      enrich_s,
            "total_s":       trie_build_s + generation_s + enrich_s,
            "prompt_tokens": prompt_tokens,
            "context_block": context_block,
        }