import networkx as nx
from gcr.trie import ProcessTrie
import torch

def extract_paths_ocel(graph: nx.DiGraph, start_node: str, max_depth: int = 6):
    """
    Extract heterogeneous OCEL paths such as:
    Event ->(relates_to)-> Object ->(related_to)-> Object ->(NEXT_FOR_...)-> Event -> ...

    Returns list of paths; each path is [(node, relation, next_node), ...]
    """
    paths = []
    stack = [(start_node, [], 0, {start_node})]

    while stack:
        node, cur_path, depth, visited = stack.pop()

        if depth >= max_depth or graph.out_degree(node) == 0:
            paths.append(cur_path)
            continue

        for _, nxt, data in graph.out_edges(node, data=True):
            relation = data.get("label", "rel")
            if nxt in visited:
                # prevent cycles
                paths.append(cur_path)
                continue

            new_path = cur_path + [(node, relation, nxt)]
            new_visited = visited | {nxt}

            stack.append((nxt, new_path, depth + 1, new_visited))

    return paths

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

# src/ocel_path_sampler.py
import networkx as nx
from typing import List, Tuple

def node_semantic_label(G: nx.DiGraph, n: str) -> str:
    etype = G.nodes[n].get("entity_type", "Node")
    if etype == "Event":
        act = G.nodes[n].get("activity", "UnknownActivity").replace(" ", "_")
        return f"Event:{act}"
    if etype == "Object":
        otyp = G.nodes[n].get("object_type", "Object").replace(" ", "_")
        return f"Object:{otyp}"
    return etype

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

def linearize_path(G: nx.DiGraph, path_triplets: List[Tuple[str, str, str]]) -> str:
    parts = []
    for u, rel, v in path_triplets:
        parts.append(node_semantic_label(G, u))
        parts.append(rel)
        parts.append(node_semantic_label(G, v))
    return " ".join(parts)

def dump_paths_for_trie(G: nx.DiGraph, start_nodes: List[str], out_txt: str, max_depth: int = 6):
    seen = set()
    with open(out_txt, "w", encoding="utf-8") as f:
        for s in start_nodes:
            for trip_path in extract_paths(G, s, max_depth=max_depth):
                sline = linearize_path(G, trip_path)
                if sline not in seen:
                    seen.add(sline)
                    f.write(sline + "\n")

def build_trie_from_ocel(graph, start_node, tokenizer, max_depth=6):
    trie = ProcessTrie()
    
    paths = extract_paths_ocel(graph, start_node, max_depth=max_depth)
    
    for path in paths:
        txt = linearize_path(path, graph)
        toks = tokenizer.encode(txt)
        trie.insert(toks)
        
    return trie

def constrained_decode(model, tokenizer, trie, max_new_tokens=40):
    prefix_tokens = []
    generated = []

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([prefix_tokens], dtype=torch.long)
        logits = model(input_ids).logits[:, -1, :]  # vocab distribution

        allowed = trie.allowed_next(prefix_tokens)
        if not allowed:
            break

        # mask
        mask = torch.full_like(logits, float('-inf'))
        for tok in allowed:
            if tok < logits.shape[-1]:
                mask[0, tok] = logits[0, tok]

        next_tok = torch.argmax(mask, dim=-1).item()
        prefix_tokens.append(next_tok)
        generated.append(next_tok)

    return tokenizer.decode(generated)