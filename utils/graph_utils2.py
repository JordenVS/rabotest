import pm4py
import pandas as pd
import networkx as nx

def build_process_graphs_ocel2(input_file_path):
    """
    Builds:
      1. G_behavior: event-only graph for path generation / Trie construction
      2. G_context: full context graph with events, objects, and structural relations

    Returns:
      G_behavior, G_context
    """

    print(f"Loading OCEL 2.0 log: {input_file_path}")
    ocel = pm4py.read_ocel2(input_file_path)

    events_df = ocel.events
    objects_df = ocel.objects
    relations_df = ocel.relations

    # --------------------------------------------------
    # Graph A: BEHAVIOR GRAPH (Event → Event only)
    # --------------------------------------------------
    G_behavior = nx.DiGraph()
    print("Building behavioral graph (events only)...")

    # --- Add Event nodes ---
    for _, row in events_df.iterrows():
        evt_id = row["ocel:eid"]
        G_behavior.add_node(
            evt_id,
            entity_type="Event",
            activity=row["ocel:activity"],
            timestamp=str(row["ocel:timestamp"]),
        )

    # --- Explicit Event-to-Event relations (if present) ---
    if ocel.e2e is not None and not ocel.e2e.empty:
        for _, row in ocel.e2e.iterrows():
            G_behavior.add_edge(
                row["ocel:eid"],
                row["ocel:eid_2"],
                label=row.get("ocel:qualifier", "EXPLICIT_FOLLOWS"),
                edge_type="behavior",
            )

    # --- Lifecycle flow edges (derived NEXT_FOR edges) ---
    events_df_sorted = events_df.sort_values("ocel:timestamp")

    grouped = relations_df.groupby("ocel:oid")

    lifecycle_edge_count = 0

    for obj_id, group in grouped:
        related_events = group["ocel:eid"].unique()
        subset = events_df_sorted[
            events_df_sorted["ocel:eid"].isin(related_events)
        ]

        event_ids = subset["ocel:eid"].tolist()
        obj_type = (
            objects_df.loc[
                objects_df["ocel:oid"] == obj_id, "ocel:type"
            ]
            .values[0]
            if obj_id in objects_df["ocel:oid"].values
            else "object"
        )

        for i in range(len(event_ids) - 1):
            G_behavior.add_edge(
                event_ids[i],
                event_ids[i + 1],
                label=f"NEXT_FOR_{obj_type}",
                edge_type="behavior",
                object_type=obj_type,
            )
            lifecycle_edge_count += 1

    print(f"Added {lifecycle_edge_count} lifecycle edges.")

    # --------------------------------------------------
    # Graph B: CONTEXT GRAPH (Events + Objects)
    # --------------------------------------------------
    G_context = nx.MultiDiGraph()
    print("Building context graph (events + objects)...")

    # --- Add Object nodes ---
    for _, row in objects_df.iterrows():
        obj_id = row["ocel:oid"]
        attrs = {
            "entity_type": "Object",
            "object_type": row["ocel:type"],
        }
        for k, v in row.items():
            if pd.notna(v) and k not in ["ocel:oid", "ocel:type"]:
                attrs[k.replace("ocel:", "")] = str(v)

        G_context.add_node(obj_id, **attrs)

    # --- Add Event nodes (again, but now with full context) ---
    for _, row in events_df.iterrows():
        evt_id = row["ocel:eid"]
        attrs = {
            "entity_type": "Event",
            "activity": row["ocel:activity"],
            "timestamp": str(row["ocel:timestamp"]),
        }
        for k, v in row.items():
            if pd.notna(v) and k not in [
                "ocel:eid",
                "ocel:activity",
                "ocel:timestamp",
            ]:
                attrs[k.replace("ocel:", "")] = str(v)

        G_context.add_node(evt_id, **attrs)

    # --- Event ↔ Object (participation) ---
    for _, row in relations_df.iterrows():
        e = row["ocel:eid"]
        o = row["ocel:oid"]
        label = row.get("ocel:qualifier", "relates_to")

        G_context.add_edge(
            e, o, label=label, edge_type="participation"
        )
        G_context.add_edge(
            o, e, label=label, edge_type="participation"
        )

    # --- Object ↔ Object (structure) ---
    if ocel.o2o is not None and not ocel.o2o.empty:
        for _, row in ocel.o2o.iterrows():
            o1 = row["ocel:oid"]
            o2 = row["ocel:oid_2"]
            label = row.get("ocel:qualifier", "related_to")

            G_context.add_edge(
                o1, o2, label=label, edge_type="structure"
            )
            G_context.add_edge(
                o2, o1, label=label, edge_type="structure"
            )

    return G_behavior, G_context

def load_graphml_to_networkx(graphml_file_path) -> nx.DiGraph:
    G = nx.read_graphml(graphml_file_path) 
    print(f"Loaded Graph with {G.number_of_nodes()} nodes.")
    return G