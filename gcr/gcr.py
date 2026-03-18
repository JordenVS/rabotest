"""
gcr/gcr.py
----------
Core GCR utilities for OCEL 2.0 process-aware graphs.

Serialization contract (all functions use this format)
-------------------------------------------------------
A path over nodes [n0, n1, n2, ...] is serialized as the *chain*:

    label(n0) rel_01 label(n1) rel_12 label(n2) ...

where label() is:
    Event nodes   → "Event:<activity_with_underscores>"
    Object nodes  → "Object:<object_type_with_underscores>"

Tokens are separated by a single space.

This format is used consistently by:
  - linearize_path          (trie construction)
  - linearize_triplets / node_semantic_label  (ground-truth dataset)
  - get_constrained_trie    (alternate trie entry-point)
"""

import networkx as nx
from gcr.trie import ProcessTrie
from typing import List, Tuple
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Canonical node label  (single source of truth)
# ---------------------------------------------------------------------------

def node_label(G: nx.DiGraph, node_id: str) -> str:
    """
    Return the canonical semantic label for *node_id*.

    Format:  "Event:<Activity_With_Underscores>"
             "Object:<object_type_with_underscores>"

    Spaces in activity / object_type names are replaced with underscores so
    that the label is a single token-friendly unit.  This matches the format
    used by node_semantic_label() in generate_eval_dataset.py.
    """
    data = G.nodes[node_id]
    entity_type = data.get("entity_type", "Node")
    if entity_type == "Event":
        raw = data.get("activity", node_id)
    else:
        raw = data.get("object_type", node_id)
    return f"{entity_type}:{raw.replace(' ', '_')}"


# ---------------------------------------------------------------------------
# Path linearization  (chain format, no duplicate interior nodes)
# ---------------------------------------------------------------------------

def linearize_path(
    path: List[Tuple[str, str, str]],
    graph: nx.DiGraph,
    sep: str = " ",
) -> str:
    """
    Convert a list of (node, rel, nxt) triples into a chain string:

        label(n0) rel_01 label(n1) rel_12 label(n2) ...

    Interior nodes are emitted exactly once.  For an empty path, returns "".

    Parameters
    ----------
    path  : List of (src_id, relation_label, tgt_id) triples produced by
            extract_paths().  The relation label is already underscore-normalised
            by extract_paths().
    graph : The NetworkX DiGraph (needed to look up node attributes).
    sep   : Token separator (default single space, matching tokenizer default).
    """
    parts = []
    for i, (node, rel, nxt) in enumerate(path):
        node_type = graph.nodes[node].get("entity_type", "Node")
        node_label = graph.nodes[node].get("activity", graph.nodes[node].get("object_type", node))
        parts.append(f"{node_type}:{node_label}")
        parts.append(rel)
        if i == len(path) - 1:  # append final node only once
            nxt_type = graph.nodes[nxt].get("entity_type", "Node")
            nxt_label = graph.nodes[nxt].get("activity", graph.nodes[nxt].get("object_type", nxt))
            parts.append(f"{nxt_type}:{nxt_label}")
    return sep.join(parts)

# ---------------------------------------------------------------------------
# Path extraction
# ---------------------------------------------------------------------------

def extract_paths(
    G: nx.DiGraph,
    start_node: str,
    max_depth: int = 6,
) -> List[List[Tuple[str, str, str]]]:
    """
    DFS enumeration of all simple paths starting from *start_node* up to
    *max_depth* hops.  Cycles are broken by per-path visited sets.

    Returns a list of paths, each path being a list of (src, rel, tgt) triples.
    Relation labels have spaces normalised to underscores.
    """
    paths: List[List[Tuple[str, str, str]]] = []
    stack = [(start_node, [], 0, {start_node})]

    while stack:
        node, cur, d, visited = stack.pop()
        if d >= max_depth or G.out_degree(node) == 0:
            if cur:
                paths.append(cur)
            continue
        for _, nxt, data in G.out_edges(node, data=True):
            rel = str(data.get("label", "rel")).replace(" ", "_")
            if nxt in visited:
                if cur:
                    paths.append(cur)   # terminate on cycle, keep partial path
                continue
            stack.append((nxt, cur + [(node, rel, nxt)], d + 1, visited | {nxt}))

    return paths


# ---------------------------------------------------------------------------
# Bulk path-string collection (used by GCRProcessAgent)
# ---------------------------------------------------------------------------

def collect_unique_path_strings(
    G: nx.DiGraph,
    start_nodes: List[str],
    max_depth: int = 3,
) -> List[str]:
    """
    Enumerate and deduplicate chain-format path strings across all *start_nodes*.

    Returns
    -------
    List[str]  — unique linearised path strings, each in canonical format.
    """
    seen: set = set()
    results: List[str] = []
    for s in start_nodes:
        for trip_path in extract_paths(G, s, max_depth=max_depth):
            s_line = linearize_path(trip_path, G)
            if s_line and s_line not in seen:
                seen.add(s_line)
                results.append(s_line)
    return results


# ---------------------------------------------------------------------------
# Trie construction
# ---------------------------------------------------------------------------

def build_trie_from_path_strings(
    path_strings: List[str],
    tokenizer_name_or_obj,
) -> ProcessTrie:
    """
    Build a ProcessTrie from a list of canonical path strings.

    Parameters
    ----------
    path_strings         : Output of collect_unique_path_strings().
    tokenizer_name_or_obj: HuggingFace tokenizer instance or model-name string.
    """
    if isinstance(tokenizer_name_or_obj, str):
        tok = AutoTokenizer.from_pretrained(tokenizer_name_or_obj, use_fast=True)
    else:
        tok = tokenizer_name_or_obj

    trie = ProcessTrie()
    for text in path_strings:
        ids = tok.encode(text, add_special_tokens=False)
        if ids:
            trie.insert(ids)
    return trie


def build_trie_from_ocel(
    graph: nx.DiGraph,
    start_node: str,
    tokenizer,
    max_depth: int = 6,
) -> ProcessTrie:
    """Convenience wrapper: extract paths from *start_node*, build trie."""
    path_strings = collect_unique_path_strings(graph, [start_node], max_depth=max_depth)
    return build_trie_from_path_strings(path_strings, tokenizer)


def get_constrained_trie(
    G: nx.DiGraph,
    start_node_id: str,
    tokenizer,
    max_hops: int = 3,
) -> ProcessTrie:
    """
    Alternative entry point (used by run_gcr_audit).
    Previously used serialize_ocel_path_v2 with arrow syntax — now unified
    to use linearize_path so all trie strings are in the same format.
    """
    return build_trie_from_ocel(G, start_node_id, tokenizer, max_depth=max_hops)


# ---------------------------------------------------------------------------
# Audit helper (kept for backward compatibility)
# ---------------------------------------------------------------------------

def run_gcr_audit(G, tokenizer, model, object_id, question):
    """
    Run a single constrained-decoding query for *object_id* and return the
    decoded path string.
    """
    from transformers import LogitsProcessorList
    from gcr.processors import GCRProcessProcessor

    trie = get_constrained_trie(G, object_id, tokenizer)

    prompt = f"Audit Question: {question}\nTarget: {object_id}\nValid Process Path:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    logits_processor = LogitsProcessorList(
        [GCRProcessProcessor(trie, [prompt_len], tokenizer)]
    )
    output_ids = model.generate(
        **inputs,
        max_new_tokens=150,
        logits_processor=logits_processor,
        num_beams=5,
        num_return_sequences=1,
    )
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
