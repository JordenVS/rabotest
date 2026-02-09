
import openai
import networkx as nx
import pandas as pd
import pm4py

def ocel_to_graph_with_pm4py(input_file_path, output_file_path) -> nx.DiGraph:
    print(f"Loading OCEL 2.0 log via pm4py: {input_file_path}...")
    ocel = pm4py.read_ocel2(input_file_path)
    
    print("Data loaded. Extracting internal tables...")
    events_df = ocel.events  # Columns: ocel:eid, ocel:activity, ocel:timestamp, ...
    objects_df = ocel.objects # Columns: ocel:oid, ocel:type, ...
    relations_df = ocel.relations # Columns: ocel:eid, ocel:oid, ocel:qualifier, ...

    G = nx.DiGraph()

    print(f"Processing {len(objects_df)} objects...")
    for _, row in objects_df.iterrows():
        # pm4py prefixes standard columns with 'ocel:'
        obj_id = row["ocel:oid"]
        obj_type = row["ocel:type"]
        
        # Extract everything else as attributes
        attrs = row.to_dict()
        clean_attrs = {
            "entity_type": "Object",
            "object_type": obj_type
        }
        
        # Clean up attribute names (remove 'ocel:' prefix for cleaner Gephi labels)
        for k, v in attrs.items():
            if pd.notna(v) and k not in ["ocel:oid", "ocel:type"]:
                clean_key = k.replace("ocel:", "")
                clean_attrs[clean_key] = str(v)

        G.add_node(obj_id, **clean_attrs)

    # --- 3. Create Event Nodes ---
    print(f"Processing {len(events_df)} events...")
    # Sort by time for HOEG control flow
    events_df = events_df.sort_values(by="ocel:timestamp")
    
    for _, row in events_df.iterrows():
        evt_id = row["ocel:eid"]
        
        attrs = row.to_dict()
        clean_attrs = {
            "entity_type": "Event",
            "activity": row["ocel:activity"],
            "timestamp": str(row["ocel:timestamp"])
        }
        
        for k, v in attrs.items():
            if pd.notna(v) and k not in ["ocel:eid", "ocel:activity", "ocel:timestamp"]:
                clean_key = k.replace("ocel:", "")
                clean_attrs[clean_key] = str(v)
                
        G.add_node(evt_id, **clean_attrs)

    print(f"Processing {len(relations_df)} relationships...")
    for _, row in relations_df.iterrows():
        source = row["ocel:eid"]
        target = row["ocel:oid"]
        # Use the OCEL 2.0 qualifier as the edge label (e.g., 'quotation')
        # If qualifier is missing (NaN), default to 'relates_to'
        label = row.get("ocel:qualifier")
        if pd.isna(label):
            label = "relates_to"
            
        G.add_edge(source, target, label=label)

    print("Adding chronological control flow...")
    # Get the list of Event IDs sorted by timestamp
    sorted_event_ids = events_df["ocel:eid"].tolist()
    
    # Create 'NEXT_EVENT' edges to represent the process flow
    for i in range(len(sorted_event_ids) - 1):
        curr_id = sorted_event_ids[i]
        next_id = sorted_event_ids[i+1]
        
        # Add edge with label (Crucial for the LLM to understand order)
        G.add_edge(curr_id, next_id, label="NEXT_EVENT")

    # --- 6. Export ---
    print(f"Exporting to {output_file_path}...")
    nx.write_graphml(G, output_file_path)
    print("Success! Import this file into Gephi.")
    return G

def load_graphml_to_networkx(graphml_file_path) -> nx.DiGraph:
    G = nx.read_graphml(graphml_file_path) 
    print(f"Loaded Graph with {G.number_of_nodes()} nodes.")
    return G

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