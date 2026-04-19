"""
graphrag/graphrag.py
---------------------
GraphRAG local-search baseline for P2P OCEL process logs.

Performs a 1-hop subgraph retrieval around a seed entity, formats it as
structured context, and passes it to an LLM for answer generation.

LLM backends supported (mirrors p2prag.py):
  - "hf"     : any HuggingFace causal LM via transformers pipeline (default)
  - "openai" : OpenAI chat models via the openai SDK

Usage example
-------------
    from graphrag.graphrag import build_graphrag_llm, perform_local_search

    # Build once, reuse across many queries
    llm = build_graphrag_llm(backend="hf", model="Qwen/Qwen2.5-7B-Instruct")

    response = perform_local_search(graph, "event:52", "What happens next?", llm=llm)
    print(response)
"""

from __future__ import annotations

from typing import Literal, Optional

import networkx as nx

LLMBackend = Literal["hf", "openai"]

# ---------------------------------------------------------------------------
# LLM factory (shared pattern with p2prag._build_llm)
# ---------------------------------------------------------------------------

def build_graphrag_llm(
    backend: LLMBackend = "hf",
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens: int = 512,
    device_map: str = "auto",
):
    if backend == "hf":
        from transformers import pipeline as hf_pipeline

        print(f"[GraphRAG] Loading HuggingFace model: {model}")
        pipe = hf_pipeline(
            "text-generation",
            model=model,
            max_new_tokens=max_new_tokens,
            device_map=device_map,
        )

        def _hf_generate(prompt: str) -> tuple[str, dict]:
            output = pipe(prompt)
            generated = output[0]["generated_text"]
            if generated.startswith(prompt):
                generated = generated[len(prompt):]
            # HF pipelines don't expose token counts natively;
            # return None so aggregate() skips these via the NaN guard
            token_meta = {
                "prompt_tokens_answer": None,
                "completion_tokens":    None,
            }
            return generated.strip(), token_meta

        print("[GraphRAG] HuggingFace model ready.")
        return _hf_generate

    if backend == "openai":
        import openai

        print(f"[GraphRAG] Using OpenAI model: {model}")

        def _openai_generate(prompt: str) -> tuple[str, dict]:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",   "content": prompt},
                ]
            )
            token_meta = {
                "prompt_tokens_answer": response.usage.prompt_tokens,
                "completion_tokens":    response.usage.completion_tokens,
            }
            return response.choices[0].message.content or "", token_meta

        return _openai_generate

    raise ValueError(
        f"Unknown LLM backend '{backend}'. Choose from: 'hf' | 'openai'"
    )
# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(graph: nx.DiGraph, entity_id: str, max_hops: int = 1) -> str:
    """
    Build a structured text context from the n-hop neighbourhood of
    *entity_id* in *graph* using BFS. Returns an empty string if the
    node is absent.

    Parameters
    ----------
    max_hops : int
        Neighbourhood depth. 1 = direct neighbours only (original behaviour),
        2-3 = wider process context at the cost of more tokens.
    """
    if entity_id not in graph.nodes:
        return ""

    lines: list[str] = []

    # Focus node
    node_attrs = graph.nodes[entity_id]
    lines.append(f"FOCUS ENTITY: {entity_id}")
    lines.append(f" - Type: {node_attrs.get('entity_type')}")
    for k, v in node_attrs.items():
        if k not in ("entity_type", "label"):
            lines.append(f" - {k}: {v}")

    # BFS up to max_hops
    visited = {entity_id}
    # queue entries: (node_id, depth)
    queue = [(entity_id, 0)]

    while queue:
        current, depth = queue.pop(0)
        if depth >= max_hops:
            continue

        neighbors = list(graph.neighbors(current))
        if not neighbors:
            continue

        hop_label = "DIRECTLY LINKED" if depth == 0 else f"HOP-{depth + 1} ENTITIES"
        lines.append(f"\n{hop_label} (from {current}):")

        for neighbor in neighbors:
            edge_data = graph.get_edge_data(current, neighbor)
            neighbor_attrs = graph.nodes[neighbor]
            rel_type = edge_data.get("label", "related_to")

            if rel_type == "NEXT_EVENT":
                act = neighbor_attrs.get("activity", "Unknown")
                ts  = neighbor_attrs.get("timestamp", "")
                lines.append(
                    f" -> [NEXT_EVENT] -> {neighbor} (Activity: '{act}' at {ts})"
                )
            else:
                obj_type = neighbor_attrs.get("object_type", "Object")
                lines.append(
                    f" -> [{rel_type}] -> {neighbor} (Type: {obj_type})"
                )

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main search function
# ---------------------------------------------------------------------------

_GRAPHRAG_PROMPT_TEMPLATE = """\
You are a process mining assistant specialising in Procure-to-Pay (P2P) event logs.
Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so explicitly.
Keep your answer concise - one or two sentences.

GRAPH CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def perform_local_search(
    graph: nx.DiGraph,
    entity_id: str,
    user_query: str,
    llm=None,
    backend: LLMBackend = "hf",
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    max_hops: int = 1, 
) -> str:
    """
    Perform a GraphRAG local search on *entity_id* and return the LLM answer.

    The function accepts either a pre-built *llm* callable (recommended when
    running many queries to avoid reloading the model) or will build one
    on-the-fly from *backend* and *model*.

    Parameters
    ----------
    graph:
        The OCEL process graph (NetworkX DiGraph).
    entity_id:
        Seed node for 1-hop context retrieval (e.g. "event:52").
    user_query:
        The natural-language question to answer.
    llm:
        Optional pre-built LLM callable from build_graphrag_llm().
        If None, one is built using *backend* and *model*.
    backend:
        LLM backend to use if *llm* is None ("hf" or "openai").
    model:
        Model identifier to use if *llm* is None.

    Returns
    -------
    The LLM's answer as a plain string.
    """
    if entity_id not in graph.nodes:
        return f"Error: Entity '{entity_id}' not found in the graph."

    context = _build_context(graph, entity_id, max_hops)

    prompt = _GRAPHRAG_PROMPT_TEMPLATE.format(
        context=context,
        question=user_query,
    )

    if llm is None:
        llm = build_graphrag_llm(backend=backend, model=model)

    answer, token_meta = llm(prompt)
    return answer, token_meta, context