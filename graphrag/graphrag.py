import networkx as nx
import openai

def perform_local_search(graph, entity_id, user_query):
    """
    Performs a Local Search (GraphRAG) on a specific node.
    
    1. Retrieval: Fetches the node, its attributes, and 1-hop neighbors.
    2. Generation: uses LLM to answer based strictly on that subgraph.
    """
    
    # --- Step 1: Validation ---
    if entity_id not in graph.nodes:
        return f"Error: Entity '{entity_id}' not found in the graph."

    # --- Step 2: Build Context (The Subgraph) ---
    context_lines = []
    
    # A. The Focus Node (e.g., The Event itself)
    node_attrs = graph.nodes[entity_id]
    context_lines.append(f"FOCUS ENTITY: {entity_id}")
    context_lines.append(f" - Type: {node_attrs.get('entity_type')}")
    # Add key attributes (Activity, Timestamp, Amount, etc.)
    for k, v in node_attrs.items():
        if k not in ['entity_type', 'label']:
            context_lines.append(f" - {k}: {v}")

    # B. The Neighbors (Objects and Flow)
    context_lines.append("\nDIRECTLY LINKED ENTITIES:")
    neighbors = list(graph.neighbors(entity_id))
    
    if not neighbors:
        context_lines.append(" (No connections found)")

    for neighbor in neighbors:
        edge_data = graph.get_edge_data(entity_id, neighbor)
        neighbor_attrs = graph.nodes[neighbor]
        
        # Get edge label (e.g., "quotation", "NEXT_EVENT")
        rel_type = edge_data.get('label', 'related_to')
        
        # Format based on what the neighbor is
        if rel_type == "NEXT_EVENT":
            # Special handling for flow - vital for Process RAG
            act = neighbor_attrs.get('activity', 'Unknown')
            time = neighbor_attrs.get('timestamp', '')
            context_lines.append(f" -> [NEXT_EVENT] -> {neighbor} (Activity: '{act}' at {time})")
        else:
            # Handling for Objects (e.g., "quotation")
            obj_type = neighbor_attrs.get('object_type', 'Object')
            context_lines.append(f" -> [{rel_type}] -> {neighbor} (Type: {obj_type})")
            # Optional: Include object attributes (e.g., "Amount")
            # for k,v in neighbor_attrs.items(): ...

    context_text = "\n".join(context_lines)

    # --- Step 3: Generate Answer (The LLM) ---
    prompt = f"""
    You are a Process Mining AI assistant.
    Answer the question strictly using the provided Graph Context.
    
    GRAPH CONTEXT:
    {context_text}
    
    QUESTION: {user_query}
    """
    
    # Simulating LLM Call (Replace with actual OpenAI/Ollama call)
    print("----- PROMPT SENT TO LLM -----")
    print(prompt)
    print("------------------------------")
    
    return openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )   