import pm4py
import pandas as pd
from llama_index.core import Document

def get_docs_from_pm4py(input_file_path):
    print(f"Loading OCEL 2.0 log via pm4py: {input_file_path}...")
    ocel = pm4py.read_ocel2(input_file_path)
    
    events_df = ocel.events
    objects_df = ocel.objects
    relations_df = ocel.relations
    o2o_df = ocel.o2o if ocel.o2o is not None else pd.DataFrame()
    e2e_df = ocel.e2e if ocel.e2e is not None else pd.DataFrame()

    rag_docs = []

    # --- 1. Process OBJECTS ---
    for _, row in objects_df.iterrows():
        obj_id = row["ocel:oid"]
        obj_type = row["ocel:type"]
        
        # Collect Attributes (mirrors your graph 'clean_attrs')
        attrs = []
        for k, v in row.items():
            if pd.notna(v) and k not in ["ocel:oid", "ocel:type"]:
                clean_key = k.replace("ocel:", "")
                attrs.append(f"{clean_key}={v}")
        
        # Find Object-to-Object links
        links = []
        if not o2o_df.empty:
            related = o2o_df[o2o_df["ocel:oid"] == obj_id]
            for _, r in related.iterrows():
                qual = r.get("ocel:qualifier", "related_to")
                links.append(f"Object {obj_id} --[{qual}]--> Object {r['ocel:oid_2']}")

        # Build Document
        header = f"Object {obj_type}:{obj_id}"
        attr_str = "Attributes: " + "; ".join(attrs) if attrs else ""
        link_str = "Links: " + " | ".join(links) if links else ""
        
        nl = [f"{obj_type} {obj_id} is present in the log."]
        if attrs: nl.append(f"It has attributes: {'; '.join(attrs)}.")
        if links: nl.extend([f"It relates to: {l}" for l in links])

        page_content = f"{header}\n{attr_str}\n{link_str}\n\n" + "\n".join(nl)
        
        rag_docs.append(Document(
            text=page_content,
            doc_id=obj_id,
            metadata={"id": obj_id, "type": obj_type, "entity_type": "Object"}
        ))

    # --- 2. Process EVENTS ---
    # Sort for lifecycle tracking
    events_df = events_df.sort_values(by="ocel:timestamp")
    
    for _, row in events_df.iterrows():
        evt_id = row["ocel:eid"]
        activity = row["ocel:activity"]
        timestamp = str(row["ocel:timestamp"])

        # Attributes
        attrs = []
        for k, v in row.items():
            if pd.notna(v) and k not in ["ocel:eid", "ocel:activity", "ocel:timestamp"]:
                clean_key = k.replace("ocel:", "")
                attrs.append(f"{clean_key}={v}")

        # E2O Relationships
        links = []
        rel_subset = relations_df[relations_df["ocel:eid"] == evt_id]
        for _, r in rel_subset.iterrows():
            qual = r.get("ocel:qualifier", "relates_to")
            # We look up object type from objects_df for the link string
            o_type = objects_df[objects_df["ocel:oid"] == r["ocel:oid"]]["ocel:type"].values[0]
            links.append(f"Event {evt_id} --[{qual}]--> {o_type}:{r['ocel:oid']}")

        # E2E Explicit Flow
        if not e2e_df.empty:
            e_rels = e2e_df[e2e_df["ocel:eid"] == evt_id]
            for _, er in e_rels.iterrows():
                qual = er.get("ocel:qualifier", "explicitly_follows")
                links.append(f"Event {evt_id} --[{qual}]--> Event {er['ocel:eid_2']}")

        # Lifecycle Flow (NEXT_FOR_object_type)
        # This mirrors your loop that calculates Event-to-Event sequence per object
        for _, r in rel_subset.iterrows():
            oid = r["ocel:oid"]
            o_type = objects_df[objects_df["ocel:oid"] == oid]["ocel:type"].values[0]
            
            # Find the next event for this specific object
            obj_events = relations_df[relations_df["ocel:oid"] == oid]["ocel:eid"].tolist()
            subset = events_df[events_df["ocel:eid"].isin(obj_events)].sort_values("ocel:timestamp")
            sorted_ids = subset["ocel:eid"].tolist()
            
            if evt_id in sorted_ids:
                idx = sorted_ids.index(evt_id)
                if idx < len(sorted_ids) - 1:
                    next_id = sorted_ids[idx+1]
                    links.append(f"Event {evt_id} --[NEXT_FOR_{o_type}]--> Event {next_id}")

        # Build Document
        header = f"Event {evt_id} | Activity: {activity} | Timestamp: {timestamp}"
        attr_str = "Attributes: " + "; ".join(attrs) if attrs else ""
        link_str = "Links: " + " | ".join(links) if links else ""
        
        nl = [f"Event {evt_id} executes activity **{activity}** at **{timestamp}**."]
        if attrs: nl.append(f"It has attributes: {'; '.join(attrs)}.")
        if links: nl.extend([f"Context: {l}" for l in links])

        page_content = f"{header}\n{attr_str}\n{link_str}\n\n" + "\n".join(nl)

        rag_docs.append(Document(
            text=page_content,
            doc_id=evt_id,
            metadata={"id": evt_id, "activity": activity, "entity_type": "Event"}
        ))

    return rag_docs