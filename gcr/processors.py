import torch
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from typing import List
from .gcr import collect_unique_path_strings, build_trie_from_path_strings

# 1. THE CONSTRAINED PROCESSOR (GCR CORE)
class GCRProcessProcessor(torch.nn.Module):
    def __init__(self, trie, prompt_lens: List[int], tokenizer):
        super().__init__()
        self.trie = trie
        self.prompt_lens = prompt_lens
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        
        for b in range(batch_size):
            # FIX: Handle Beam Search flattening. 
            # If we have 1 prompt but 3 beams, use prompt_lens[0] for all beams.
            # If we have multiple prompts in a batch, we map b back to the original index.
            prompt_idx = b // (batch_size // len(self.prompt_lens))
            p_len = self.prompt_lens[prompt_idx]
            
            current_gen_ids = input_ids[b][p_len:].tolist()
            allowed = self.trie.allowed_next(current_gen_ids)

            if allowed:
                mask = torch.full_like(scores[b], float("-inf"))
                allowed_tensor = torch.tensor(list(allowed), dtype=torch.long, device=scores.device)
                mask[allowed_tensor] = scores[b, allowed_tensor]
                scores[b] = mask
            else:
                # Allow EOS to stop the path if the Trie is exhausted
                mask = torch.full_like(scores[b], float("-inf"))
                mask[self.tokenizer.eos_token_id] = 0
                scores[b] = mask
                
        return scores
# 2. THE ENGINE: FROM LINKED ENTITY TO PATHS
class GCRProcessAgent:
    def __init__(self, model_id, graph: nx.DiGraph):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float32, 
            low_cpu_mem_usage=True,
            device_map="cpu" 
        )
        self.device = "cpu"
        self.graph = graph

    def _build_trie_for_seed(self, seed_node: str, max_depth: int):
        """
        GCR specific: We build a local trie of valid future paths 
        from the current OCEL state
        """
        
        paths = collect_unique_path_strings(self.graph, [seed_node], max_depth=max_depth)
       # ser_paths = serialize_ocel_paths_v2(self.graph, paths) TODO: Fix serialization.
        # Ensure we add a specific 'Path Start' anchor if needed
        return build_trie_from_path_strings(paths, self.tokenizer)

    def generate_compliant_paths(self, seed_entity: str, question: str, num_paths=3):
        # A. Setup the Prompt
        # We tell the LLM to start the path with a specific prefix for alignment
        prompt = f"Question: {question}\nContext: Found Object {seed_entity}.\nReasoning Path: "
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        # B. Build the Constraints
        # This is where GCR differs: the Trie is seeded by the graph's neighborhood
        process_trie = self._build_trie_for_seed(seed_entity, max_depth=4)
        
        # C. Initialize GCR Processor
        # We pass the prompt length so it knows where 'Reasoning Path: ' ends
        logits_processor = LogitsProcessorList([
            GCRProcessProcessor(process_trie, [prompt_len], self.tokenizer)
        ])

        # D. Constrained Beam Search
        # This forces the LLM to explore the top-K valid paths in the OCEL logs
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=num_paths,
            num_return_sequences=num_paths,
            logits_processor=logits_processor,
            early_stopping=True
        )

        # E. Decode Results
        paths = [self.tokenizer.decode(g[prompt_len:], skip_special_tokens=True) for g in output_ids]
        return paths

# graph = load_ocel_to_networkx("logs.jsonocel")
# agent = GCRProcessAgent("meta-llama/Llama-3-8b", graph)
# results = agent.generate_compliant_paths("Order_99", "What happens after payment?")