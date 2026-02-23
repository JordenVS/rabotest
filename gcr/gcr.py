import networkx as nx
from gcr.trie import ProcessTrie
from typing import List, Tuple
from transformers import AutoTokenizer

def linearize_path(path, graph, sep=" "):
    """
    Convert [(node, rel, nxt), ...] into:
    "Event:CreatePO NEXT_FOR_purchase_order Event:ApprovePO ..."
    """
    parts = []

    for node, rel, nxt in path:
        # Describe the current node
        node_type = graph.nodes[node].get("entity_type", "Node")
        node_label = graph.nodes[node].get("activity", graph.nodes[node].get("object_type", ""))
        node_str = f"{node_type}:{node_label}" if node_label else f"{node_type}:{node}"

        # Describe the next node
        nxt_type = graph.nodes[nxt].get("entity_type", "Node")
        nxt_label = graph.nodes[nxt].get("activity", graph.nodes[nxt].get("object_type", ""))
        nxt_str = f"{nxt_type}:{nxt_label}" if nxt_label else f"{nxt_type}:{nxt}"

        parts.append(node_str)
        parts.append(rel)
        parts.append(nxt_str)

    return sep.join(parts)

def extract_paths(G: nx.DiGraph, start_node: str, max_depth: int = 6) -> List[List[Tuple[str, str, str]]]:
    paths, stack = [], [(start_node, [], 0, {start_node})]
    while stack:
        node, cur, d, visited = stack.pop()
        if d >= max_depth or G.out_degree(node) == 0:
            if cur: paths.append(cur)
            continue
        for _, nxt, data in G.out_edges(node, data=True):
            rel = str(data.get("label", "rel")).replace(" ", "_")
            if nxt in visited:
                if cur: paths.append(cur)  # terminate on cycle
                continue
            stack.append((nxt, cur + [(node, rel, nxt)], d+1, visited | {nxt}))
    return paths


def collect_unique_path_strings(G: nx.DiGraph, start_nodes: List[str], max_depth: int = 3) -> List[str]:    
    """    Deduplicate path strings across many anchors.    """    
    seen = set()    
    results = []    
    for s in start_nodes:        
        for trip_path in extract_paths(G, s, max_depth=max_depth):            
            sline = linearize_path(G, trip_path)            
            if sline not in seen:                
                seen.add(sline)                
                results.append(sline)    
    return results

def build_trie_from_ocel(graph, start_node, tokenizer, max_depth=6):
    trie = ProcessTrie()
    
    paths = extract_paths(graph, start_node, max_depth=max_depth)
    
    for path in paths:
        txt = linearize_path(path, graph)
        toks = tokenizer.encode(txt)
        trie.insert(toks)
        
    return trie

def build_trie_from_path_strings(path_strings: List[str], tokenizer_name_or_obj) -> ProcessTrie:
    """
    path_strings: list of semantic path strings (no IDs).
    tokenizer_name_or_obj: either a string (model name) or an instantiated tokenizer.
    """
    if isinstance(tokenizer_name_or_obj, str):
        tok = AutoTokenizer.from_pretrained(tokenizer_name_or_obj, use_fast=True)
    else:
        tok = tokenizer_name_or_obj

    trie = ProcessTrie()
    for text in path_strings:
        # Tokenize exactly as you will at inference time
        ids = tok.encode(text, add_special_tokens=False)
        if ids:
            trie.insert(ids)

    return trie