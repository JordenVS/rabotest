# PA-GCR: Process-Aware Graph-Constrained Reasoning

[cite_start]**PA-GCR** is a framework designed to eliminate hallucinations in Large Language Models (LLMs) when reasoning over process data. [cite_start]By grounding the model in **Object-Centric Event Logs (OCEL 2.0)**, it ensures that every generated reasoning step is behaviorally valid according to the actual process data[cite: 762, 792].

## Key Features
* [cite_start]**Hard Constraints:** Uses a **Process Trie** (prefix-tree) to restrict token-level decoding to structurally valid paths during the LLM's beam search.
* [cite_start]**Dual-Graph Architecture:** Separates behavioral logic ($G_{behavior}$) from contextual details ($G_{context}$) to enable precise structural enforcement combined with rich attribute injection.
* [cite_start]**Efficiency:** The Process Trie prunes the decoding search space, which not only ensures validity but also improves computational efficiency by reducing path generation time.

## Workflow
1.  [cite_start]**OCEL Data Downloading** Loads the OCEL 2.0 log data from Zenodo.
2.  [cite_start]**Graph Construction:** Transforms OCEL 2.0 data into a dual-graph representation (behavioral and context graphs).
2.  [cite_start]**Path Enumeration & Trie Indexing:** Identifies object-valid paths anchored at a specific entity and indexes them into a Process Trie.
3.  [cite_start]**Constrained Decoding:** Enforces process rules directly at the token level during generation, assigning a probability of zero to invalid transitions.
4.  [cite_start]**Context Integration:** Reifies activity sequences back into specific event/object instances to provide the factual evidence required for the final answer.

## Technical Stack
* [cite_start]**Language:** Python 
* [cite_start]**Process Mining:** `pm4py` 
* [cite_start]**Graphs:** `NetworkX` 
* [cite_start]**LLM Handling:** `Hugging Face Transformers`
* [cite_start]**Models:** * **Path Generation:** Qwen2.5-1.5B-Instruct (run locally for logit access)
    * [cite_start]**Final Answer:** GPT-4o-mini (via OpenAI API) 