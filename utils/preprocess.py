from langchain_community.document_loaders import JSONLoader
from datetime import datetime, timezone
from llama_index.core import Document
import json

jq_event_lines = (
    # Find any object with id, type, attributes
    '.. | objects | select(has("id"))'
)

def get_docs(jsonpath):
    loader = JSONLoader(
        file_path=jsonpath,
        jq_schema=jq_event_lines,
        text_content=False
    )

    def jsondic(content):
        try:        
            ev = json.loads(content)  # parse the JSON string    
        except json.JSONDecodeError:        # Not JSON? Then just return as-is.        
            return doc
        return ev

    rag_docs = []
    for doc in loader.lazy_load():
        ev = jsondic(doc.page_content)  # this is the dict: {"id", "type", "attributes", "relationships", ...}

        # Essentials
        ev_id = ev.get("id", "")
        ev_type = ev.get("type", "")

        # Keep only meaningful attribute values (ignore "", None)
        attrs = []
        for a in ev.get("attributes", []):
            v = a.get("value")
            if v not in ("", None):
                # you can also skip the epoch timestamps if you want
                attrs.append(f"{a.get('name') }={v}@{a.get('time')}")

        # Relationships (optional but useful)
        rels = []
        for r in ev.get("relationships", []):
            rels.append(f"{r.get('qualifier')}: {r.get('objectId')}")

        # Build a compact, readable line (ideal for embeddings)
        lines = [f"{ev_id} | {ev_type}"]
        if attrs:
            lines.append("; ".join(attrs))
        if rels:
            lines.append(" | ".join(rels))

        # Replace the dict in page_content with your formatted string
        doc.page_content = " | ".join(lines)

        # Optional: keep minimal, useful metadata (traceability)
        doc.metadata = {
            "id": ev_id,
            "type": ev_type,
        }

        rag_docs.append(doc)

    return rag_docs

# --- 1) jq schemas: pull events OR objects from an OCEL 2.0 JSON ---
# Adjust keys if your file nests under another root.
jq_events = '.. | .events? // empty | select(type=="array")[]'
jq_objects = '.. | .objects? // empty | select(type=="array")[]'

def _is_real_time(value: str) -> bool:
    """Filter out placeholder/epoch-like timestamps (e.g., 1970-01-01) and keep real ones."""
    if not value:
        return False
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # Treat anything before 1971 as placeholder; tweak threshold as needed
        return dt.year > 1970
    except Exception:
        return False

def _fmt_ts(value: str) -> str:
    """Return ISO 8601 in UTC if valid, else ''."""
    if not _is_real_time(value):
        return ""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()

def _kv(name: str, value, time_str: str = None) -> str:
    """Format attribute name=value [@time] compactly."""
    if value in ("", None):
        return ""
    t = _fmt_ts(time_str) if time_str else ""
    return f"{name}={value}" + (f" @{t}" if t else "")

def _clean_list(items):
    return [x for x in items if x]

def _safe_json(content):
    if isinstance(content, (dict, list)):
        return content
    try:
        return json.loads(content)
    except Exception:
        return {}

def get_docs_extensive(
    json_path: str,
    mode: str = "both",   # "event" | "object" | "both"
    explain_codes: dict = None,  # e.g., {"MSEG-BWART:101": "Standard goods receipt"}
):
    """
    Transform an OCEL 2.0 JSON into GraphRAG-friendly text documents.
    Emits short sentences with explicit entities & relations, plus a compact header.
    """
    rag_docs = []

    # --- 2) Load EVENTS ---
    if mode in ("event", "both"):
        ev_loader = JSONLoader(file_path=json_path, jq_schema=jq_events, text_content=False)
        for doc in ev_loader.lazy_load():
            ev = _safe_json(doc.page_content)  # {id, activity, timestamp, attributes, e2o, ...}
            ev_id = ev.get("id") or ev.get("event_id") or ""
            activity = ev.get("activity", "")
            ts = _fmt_ts(ev.get("timestamp", ""))

            # Attributes (name/value/time); OCEL 2.0 often stores a list of dicts
            attrs_lines = []
            for a in ev.get("attributes", []):
                line = _kv(a.get("name"), a.get("value"), a.get("time"))
                if line:
                    # Optional humanization of SAP codes
                    if explain_codes and (k := f"{a.get('name')}:{a.get('value')}") in explain_codes:
                        line += f"  ({explain_codes[k]})"
                    attrs_lines.append(line)

            # E2O relationships: event -> object (with qualifiers/roles)
            e2o_lines = []
            for r in ev.get("e2o", []) or ev.get("relationships", []):
                qual = r.get("qualifier") or r.get("role") or "related_to"
                obj_id = r.get("object_id") or r.get("objectId") or ""
                obj_type = r.get("object_type") or r.get("objectType") or ""
                if obj_id:
                    e2o_lines.append(f"{ev_id} --[{qual}]--> {obj_type or 'object'}:{obj_id}")

            # Build a compact, *and* a natural-language section
            header = _clean_list([
                f"Event {ev_id}",
                f"Activity: {activity}" if activity else "",
                f"Timestamp: {ts}" if ts else "",
            ])
            if attrs_lines:
                header.append("Attributes: " + "; ".join(attrs_lines))
            if e2o_lines:
                header.append("Links: " + " | ".join(e2o_lines))

            # Natural-language sentences to help LLM extraction
            nl = []
            subj = f"Event {ev_id}"
            if activity:
                if ts:
                    nl.append(f"{subj} executes activity **{activity}** at **{ts}**.")
                else:
                    nl.append(f"{subj} executes activity **{activity}**.")
            for r in ev.get("e2o", []) or ev.get("relationships", []):
                qual = r.get("qualifier") or r.get("role") or "related_to"
                obj_id = r.get("object_id") or r.get("objectId") or ""
                obj_type = r.get("object_type") or r.get("objectType") or "object"
                if obj_id:
                    nl.append(f"{subj} is **{qual}** **{obj_type} {obj_id}**.")
            if attrs_lines:
                nl.append(f"{subj} has attributes: " + "; ".join(attrs_lines) + ".")

            # Final page_content (compact header + sentences)
            doc.page_content = "\n".join(header) + "\n\n" + "\n".join(nl)
            doc.metadata = {"id": ev_id, "type": "event", "activity": activity, "source": "ocel.events"}
            rag_docs.append(doc)

    # --- 3) Load OBJECTS ---
    if mode in ("object", "both"):
        ob_loader = JSONLoader(file_path=json_path, jq_schema=jq_objects, text_content=False)
        for doc in ob_loader.lazy_load():
            obj = _safe_json(doc.page_content)  # {id, type, attributes, o2o, ...}
            ob_id = obj.get("id", "")
            ob_type = obj.get("type", "")

            # Attributes over time for this object
            attrs_lines = []
            for a in obj.get("attributes", []):
                line = _kv(a.get("name"), a.get("value"), a.get("time"))
                if line:
                    if explain_codes and (k := f"{a.get('name')}:{a.get('value')}") in explain_codes:
                        line += f"  ({explain_codes[k]})"
                    attrs_lines.append(line)

            # O2O relationships: object -> object (qualified)
            o2o_lines = []
            for r in obj.get("o2o", []) or obj.get("relationships", []):
                qual = r.get("qualifier") or r.get("role") or "related_to"
                tgt_id = r.get("target_object_id") or r.get("targetId") or r.get("objectId") or ""
                tgt_type = r.get("target_object_type") or r.get("targetType") or r.get("objectType") or ""
                if tgt_id:
                    o2o_lines.append(f"{ob_type}:{ob_id} --[{qual}]--> {tgt_type or 'object'}:{tgt_id}")

            header = _clean_list([
                f"Object {ob_type}:{ob_id}",
                ("Attributes: " + "; ".join(attrs_lines)) if attrs_lines else "",
                ("Links: " + " | ".join(o2o_lines)) if o2o_lines else "",
            ])

            nl = []
            subj = f"{ob_type} {ob_id}" if ob_type else f"Object {ob_id}"
            nl.append(f"{subj} is present in the log.")
            if attrs_lines:
                nl.append(f"{subj} has attributes: " + "; ".join(attrs_lines) + ".")
            for r in obj.get("o2o", []) or obj.get("relationships", []):
                qual = r.get("qualifier") or r.get("role") or "related_to"
                tgt_id = r.get("target_object_id") or r.get("targetId") or r.get("objectId") or ""
                tgt_type = r.get("target_object_type") or r.get("targetType") or r.get("objectType") or "object"
                if tgt_id:
                    nl.append(f"{subj} is **{qual}** **{tgt_type} {tgt_id}**.")

            doc.page_content = "\n".join(header) + "\n\n" + "\n".join(nl)
            doc.metadata = {"id": ob_id, "type": ob_type or "object", "source": "ocel.objects"}

    return rag_docs

def convert_to_li_document(doc):
    text = doc.page_content
    metadata = doc.metadata

    doc_id = metadata.get("id")
    return Document(text=text, doc_id=doc_id, metadata=metadata)