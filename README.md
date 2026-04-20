# PA-GCR: Process-Aware Graph-Constrained Reasoning

**PA-GCR** is a framework designed to eliminate hallucinations in Large Language Models (LLMs) when reasoning over process data. By grounding the model in **Object-Centric Event Logs (OCEL 2.0)**, it ensures that every generated reasoning step is behaviorally valid according to the actual process data.

## Key Features
* **Hard Constraints:** Uses a **Process Trie** (prefix-tree) to restrict token-level decoding to structurally valid paths during the LLM's beam search.
* **Dual-Graph Architecture:** Separates behavioral logic ($G_{behavior}$) from contextual details ($G_{context}$) to enable precise structural enforcement combined with rich attribute injection.
* **Efficiency:** The Process Trie prunes the decoding search space, which not only ensures validity but also improves computational efficiency by reducing path generation time.

## Workflow
1.  **OCEL Data Downloading** Loads the OCEL 2.0 log data from Zenodo.
2.  **Graph Construction:** Transforms OCEL 2.0 data into a dual-graph representation (behavioral and context graphs).
2.  **Path Enumeration & Trie Indexing:** Identifies object-valid paths anchored at a specific entity and indexes them into a Process Trie.
3.  **Constrained Decoding:** Enforces process rules directly at the token level during generation, assigning a probability of zero to invalid transitions.
4.  **Context Integration:** Reifies activity sequences back into specific event/object instances to provide the factual evidence required for the final answer.

## Technical Stack
* **Language:** Python 
* **Process Mining:** `pm4py` 
* **Graphs:** `NetworkX` 
* **LLM Handling:** `Hugging Face Transformers`
* **Models:** * **Path Generation:** Qwen2.5-1.5B-Instruct (run locally for logit access)
    * **Final Answer:** GPT-4o-mini (via OpenAI API) 