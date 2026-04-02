import pm4py
import pandas as pd
#from llama_index.core import Document
from langchain_core.documents import Document as LCDocument

def get_docs_from_pm4py(input_file_path):
    print(f"Loading OCEL 2.0 log via pm4py: {input_file_path}...")
    ocel = pm4py.read_ocel2(input_file_path)

    events_df = ocel.events.copy()
    objects_df = ocel.objects.copy()
    relations_df = ocel.relations.copy()

    o2o_df = ocel.o2o.copy() if ocel.o2o is not None else pd.DataFrame()
    e2e_df = ocel.e2e.copy() if ocel.e2e is not None else pd.DataFrame()

    rag_docs = []

    def clean_cols(df):
        df = df.copy()
        df.columns = [
            c.replace("ocel:", "ocel_").replace(":", "_")
            for c in df.columns
        ]
        return df

    objects_df = clean_cols(objects_df)
    events_df = clean_cols(events_df)
    relations_df = clean_cols(relations_df)
    if not o2o_df.empty:
        o2o_df = clean_cols(o2o_df)
    if not e2e_df.empty:
        e2e_df = clean_cols(e2e_df)

    oid_to_type = dict(zip(objects_df["ocel_oid"], objects_df["ocel_type"]))

    relations_by_event = relations_df.groupby("ocel_eid")
    relations_by_object = relations_df.groupby("ocel_oid")

    o2o_by_oid = o2o_df.groupby("ocel_oid") if not o2o_df.empty else {}
    e2e_by_eid = e2e_df.groupby("ocel_eid") if not e2e_df.empty else {}

    # Object → ordered event sequence
    object_event_sequences = {}
    for oid, group in relations_by_object:
        eids = group["ocel_eid"].tolist()
        subset = events_df[events_df["ocel_eid"].isin(eids)]
        sorted_ids = subset.sort_values("ocel_timestamp")["ocel_eid"].tolist()
        object_event_sequences[oid] = sorted_ids

    for row in objects_df.to_dict("records"):
        obj_id = row["ocel_oid"]
        obj_type = row["ocel_type"]

        # Attributes
        attrs = [
            f"{k.replace('ocel_', '')}={v}"
            for k, v in row.items()
            if pd.notna(v) and k not in ["ocel_oid", "ocel_type"]
        ]

        # O2O links
        links = []
        if not o2o_df.empty and obj_id in o2o_by_oid.groups:
            for r in o2o_by_oid.get_group(obj_id).to_dict("records"):
                qual = r.get("ocel_qualifier", "related_to")
                links.append(
                    f"Object {obj_id} --[{qual}]--> Object {r['ocel_oid_2']}"
                )

        # Build document
        header = f"Object {obj_type}:{obj_id}"
        attr_str = "Attributes: " + "; ".join(attrs) if attrs else ""
        link_str = "Links: " + " | ".join(links) if links else ""

        nl = [f"{obj_type} {obj_id} is present in the log."]
        if attrs:
            nl.append(f"It has attributes: {'; '.join(attrs)}.")
        if links:
            nl.extend([f"It relates to: {l}" for l in links])

        page_content = f"{header}\n{attr_str}\n{link_str}\n\n" + "\n".join(nl)

        # rag_docs.append(Document(
        #     text=page_content,
        #     doc_id=obj_id,
        #     metadata={"id": obj_id, "type": obj_type, "entity_type": "Object"}
        # ))

        rag_docs.append(LCDocument(
        page_content=page_content,
        metadata={"id": obj_id, "type": obj_type, "entity_type": "Object"}
))

    for row in events_df.to_dict("records"):
        evt_id = row["ocel_eid"]
        activity = row["ocel_activity"]
        timestamp = str(row["ocel_timestamp"])

        attrs = [
            f"{k.replace('ocel_', '')}={v}"
            for k, v in row.items()
            if pd.notna(v) and k not in ["ocel_eid", "ocel_activity", "ocel_timestamp"]
        ]

        links = []

        # E2O relations
        if evt_id in relations_by_event.groups:
            rel_subset = relations_by_event.get_group(evt_id)

            for r in rel_subset.to_dict("records"):
                oid = r["ocel_oid"]
                qual = r.get("ocel_qualifier", "relates_to")
                o_type = oid_to_type.get(oid, "Unknown")

                links.append(
                    f"Event {evt_id} --[{qual}]--> {o_type}:{oid}"
                )

                # Lifecycle optimization
                seq = object_event_sequences.get(oid, [])
                if evt_id in seq:
                    idx = seq.index(evt_id)
                    if idx < len(seq) - 1:
                        next_id = seq[idx + 1]
                        links.append(
                            f"Event {evt_id} --[NEXT_FOR_{o_type}]--> Event {next_id}"
                        )

        # E2E relations
        if not e2e_df.empty and evt_id in e2e_by_eid.groups:
            for er in e2e_by_eid.get_group(evt_id).to_dict("records"):
                qual = er.get("ocel_qualifier", "explicitly_follows")
                links.append(
                    f"Event {evt_id} --[{qual}]--> Event {er['ocel_eid_2']}"
                )

        # Build document
        header = f"Event {evt_id} | Activity: {activity} | Timestamp: {timestamp}"
        attr_str = "Attributes: " + "; ".join(attrs) if attrs else ""
        link_str = "Links: " + " | ".join(links) if links else ""

        nl = [f"Event {evt_id} executes activity **{activity}** at **{timestamp}**."]
        if attrs:
            nl.append(f"It has attributes: {'; '.join(attrs)}.")
        if links:
            nl.extend([f"Context: {l}" for l in links])

        page_content = f"{header}\n{attr_str}\n{link_str}\n\n" + "\n".join(nl)

        # rag_docs.append(Document(
        #     text=page_content,
        #     doc_id=evt_id,
        #     metadata={"id": evt_id, "activity": activity, "entity_type": "Event"}
        # ))

        rag_docs.append(LCDocument(
    page_content=page_content,
    metadata={"id": evt_id, "activity": activity, "entity_type": "Event"}
))

    print(f"Created {len(rag_docs)} documents from OCEL log.")
    return rag_docs

def to_langchain_docs(docs) -> list:
    """Convert LlamaIndex or plain Documents to LangChain Documents."""
    lc_docs = []
    print(f"Converting {len(docs)} documents to LangChain format...")
    for doc in docs:
        # LlamaIndex Document has .text, LangChain has .page_content
        text = getattr(doc, "page_content", None) or getattr(doc, "text", "")
        metadata = getattr(doc, "metadata", {})
        lc_docs.append(LCDocument(page_content=text, metadata=metadata))
    print(f"Converted {len(docs)} documents to LangChain format.")
    return lc_docs