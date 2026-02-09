import networkx as nx
import pandas as pd
import pm4py
from collections import deque
from typing import List, Dict, Any, Set


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

    # Add o2o and e2e relations if they exist 
    if ocel.o2o is not None and not ocel.o2o.empty:
        print(f"Adding {len(ocel.o2o)} Object-to-Object relations...")
        for _, row in ocel.o2o.iterrows():
            source = row["ocel:oid"]
            target = row["ocel:oid_2"]
            qualifier = row.get("ocel:qualifier", "related_to")
            
            # Add edge (e.g., Order -> Item)
            G.add_edge(source, target, label=qualifier)

    # --- 3b. Event-to-Event (Explicit System Flow) ---
    # If the log contains this, it trumps timestamp sorting
    if ocel.e2e is not None and not ocel.e2e.empty:
        print(f"Adding {len(ocel.e2e)} explicit Event-to-Event relations...")
        for _, row in ocel.e2e.iterrows():
            source = row["ocel:eid"]
            target = row["ocel:eid_2"]
            qualifier = row.get("ocel:qualifier", "explicitly_follows")
            
            # Add edge
            G.add_edge(source, target, label=qualifier)
    else:
        print("No explicit E2E table found.")

    # We group relations by Object ID to get the specific history of *that* object
    grouped = relations_df.groupby("ocel:oid")
    
    count = 0
    for obj_id, group in grouped:
        # Get all events for this specific object
        related_event_ids = group["ocel:eid"].unique()
        
        # Get the event data subset and sort by time
        subset = events_df[events_df["ocel:eid"].isin(related_event_ids)]
        subset = subset.sort_values("ocel:timestamp")
        
        sorted_ids = subset["ocel:eid"].tolist()
        
        # Get the object type (e.g., "purchase_order") for the edge label
        # We look it up from the graph node we created earlier
        obj_type = G.nodes[obj_id].get("object_type", "object")
        
        # Create the chain: Event A -> Event B
        for i in range(len(sorted_ids) - 1):
            u = sorted_ids[i]
            v = sorted_ids[i+1]
            
            # Label example: "NEXT_FOR_purchase_order"
            # This is extremely helpful for the LLM to understand context
            edge_label = f"NEXT_FOR_{obj_type}"
            
            # We use a MultiGraph concept by adding a unique key if needed, 
            # but standard DiGraph overwrites duplicate edges. 
            # For RAG, simple connectivity is usually enough.
            G.add_edge(u, v, label=edge_label)
            count += 1
            
    print(f"Added {count} lifecycle flow edges.")

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

# def ground_entities(G, entities):
#     grounded = {
#         "event_nodes": [],
#         "object_nodes": []
#     }

#     for n, data in G.nodes(data=True):
#         if n.startswith("event:"):
#             if data.get("activity", "").lower() in entities["activities"]:
#                 grounded["event_nodes"].append(n)

#         else:  # object node
#             if n in entities["object_instances"]:
#                 grounded["object_nodes"].append(n)
#             else:
#                 obj_type = n.split(":")[0].lower()
#                 if obj_type in entities["object_types"]:
#                     grounded["object_nodes"].append(n)

#     return grounded

def resolve_anchor_nodes(G, ner):
    """
    Extract anchor nodes directly from the NER output.
    No entity linking layer. Node IDs are already in OCEL format:
      event:<eid>
      <object_type>:<oid>
    """
    anchors = set()

    # 1. Direct matches: event:<eid> and object_type:<oid>
    for inst in ner.get("event_instances", []):
        if inst in G:
            anchors.add(inst)

    for inst in ner.get("object_instances", []):
        if inst in G:
            anchors.add(inst)

    # 2. Object types: include all nodes with prefix "<type>:"
    for otype in ner.get("object_types", []):
        prefix = f"{otype.lower()}:"
        for n in G.nodes:
            if n.lower().startswith(prefix):
                anchors.add(n)

    # 3. Activities: include all event nodes whose activity matches
    for act in ner.get("activities", []):
        target = act.lower()
        for n, data in G.nodes(data=True):
            if data.get("entity_type") == "Event":
                if data.get("activity", "").lower() == target:
                    anchors.add(n)

    return anchors

def enumerate_paths_unconstrained(
    G: nx.DiGraph,
    linked: Dict[str, Any],
    *,
    max_depth: int = 2,
    max_paths: int = 20
) -> List[List[str]]:
    """
    Enumerate node-paths reachable from anchors without applying semantic constraints.
    Only bounds are max_depth and max_paths, and cycle-avoidance within each path.
    """
    anchors = resolve_anchor_nodes(linked)
    if not anchors:
        return []

    paths: List[List[str]] = []
    for anchor in anchors:
        if anchor not in G:
            continue

        # BFS queue holds (current_node, path_so_far)
        q = deque([(anchor, [anchor])])

        while q and len(paths) < max_paths:
            node, path = q.popleft()

            if len(path) > 1:
                # Collect all non-trivial paths
                paths.append(path)
                if len(paths) >= max_paths:
                    break

            # Stop expanding beyond max_depth
            if len(path) >= max_depth:
                continue

            for nbr in G.successors(node):
                if nbr not in path:           # simple cycle avoidance
                    q.append((nbr, path + [nbr]))

        if len(paths) >= max_paths:
            break

    return paths