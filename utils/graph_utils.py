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

def build_vocabularies_from_local_graph(G):
    activities = set()
    object_types = set()
    qualifiers = set()

    for _, data in G.nodes(data=True):
        if data.get("entity_type") == "Event":
            activities.add(data["activity"].lower())
        elif data.get("entity_type") == "Object":
            object_types.add(data["object_type"].lower())

    for _, _, edata in G.edges(data=True):
        qualifiers.add(edata.get("label", "").lower())

    return activities, object_types, qualifiers

def ground_entities(G, entities):
    grounded = {
        "event_nodes": [],
        "object_nodes": []
    }

    for n, data in G.nodes(data=True):
        if n.startswith("event:"):
            if data.get("activity", "").lower() in entities["activities"]:
                grounded["event_nodes"].append(n)

        else:  # object node
            if n in entities["object_instances"]:
                grounded["object_nodes"].append(n)
            else:
                obj_type = n.split(":")[0].lower()
                if obj_type in entities["object_types"]:
                    grounded["object_nodes"].append(n)

    return grounded

