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
            sline = linearize_path(trip_path, G)            
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

def serialize_ocel_path(G, path_nodes):
    """
    Converts a list of node IDs from your graph into a GCR-standard string.
    Example: Event:Create_PO -> Object:Order_123 -> Event:Approve_PO
    """
    serialized_parts = []
    for i in range(len(path_nodes)):
        node_id = path_nodes[i]
        attr = G.nodes[node_id]
        
        if attr["entity_type"] == "Event":
            label = f"Event:{attr['activity']}"
        else:
            label = f"Object:{attr['object_type']}" # We use type for general paths
        
        serialized_parts.append(label)
        
        # Add the relationship label if there's a next node
        if i < len(path_nodes) - 1:
            edge_data = G.get_edge_data(node_id, path_nodes[i+1])
            rel = edge_data.get("label", "relates_to")
            serialized_parts.append(f" [{rel}] ")
            
    return "".join(serialized_parts)

def serialize_ocel_path_v2(G, path_nodes):
    parts = []
    for i in range(len(path_nodes)):
        curr_node = path_nodes[i]
        attr = G.nodes[curr_node]
        
        # 1. Add the Node with a clear Prefix
        if attr["entity_type"] == "Event":
            parts.append(f"Event:{attr['activity']}")
        else:
            parts.append(f"Object:{attr['object_type']}")
            
        # 2. Add the Edge/Relation ONLY if there is a next node
        if i < len(path_nodes) - 1:
            next_node = path_nodes[i+1]
            edge_data = G.get_edge_data(curr_node, next_node)
            # If MultiDiGraph, edge_data might be a dict of dicts
            rel_label = edge_data.get("label", "related_to")
            
            # Use arrows and brackets to give the LLM structural cues
            parts.append(f" --({rel_label})--> ")
            
    return "".join(parts)

def get_constrained_trie(G, start_node_id, tokenizer, max_hops=3):
    # 1. Find all paths in the graph starting from the seed (e.g., Order_123)
    # This uses your existing extract_paths logic
    raw_paths = extract_paths(G, start_node_id, max_depth=max_hops)
    
    trie = ProcessTrie()
    for p in raw_paths:
        # Convert path (list of triples) to flat list of nodes
        node_sequence = [p[0][0]] + [step[2] for step in p]
        
        # 2. Serialize to string
        path_str = serialize_ocel_path_v2(G, node_sequence)
        
        # 3. Tokenize with the leading space (GCR requirement)
        token_ids = tokenizer.encode(" " + path_str, add_special_tokens=False)
        
        # 4. Insert into Trie
        trie.insert(token_ids)
        # Allow the LLM to stop at any valid event completion
        trie.insert(token_ids + [tokenizer.eos_token_id])
        
    return trie

def run_gcr_audit(G, tokenizer, model, object_id, question):
    # 1. Build the Trie for this specific Object ID
    # This acts as the 'allowed' map for the LLM
    trie = get_constrained_trie(G, object_id, tokenizer)
    
    # 2. Prepare the Prompt
    prompt = f"Audit Question: {question}\nTarget: {object_id}\nValid Process Path:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    
    # 3. Setup the GCR Logits Processor
    # (Using the GCRProcessProcessor class we drafted earlier)
    from .processors import GCRProcessProcessor
    logits_processor = LogitsProcessorList([
        GCRProcessProcessor(trie, [prompt_len], tokenizer)
    ])
    
    # 4. Generate (Constrained)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=150,
        logits_processor=logits_processor,
        num_beams=5,
        num_return_sequences=1 # Or more if you want multiple valid paths
    )
    
    # 5. Decode
    full_path = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    return full_path