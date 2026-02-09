import re
from typing import Dict, Any, List

def normalize(term: str) -> str:
    return term.lower().replace("_", " ").strip()

def make_regex(term: str) -> str:
    # Escape any special chars, preserve spaces
    t = re.escape(term)
    return rf"\b{t}\b"

def extract_entities_ocel(
    query: str,
    activities: List[str],
    object_types: List[str]
) -> Dict[str, Any]:
    """
    Extract event instances, object instances, object types, activities,
    and high-level constraints from a user query for OCEL-based logs.
    """

    q = query
    q_low = query.lower()

    # Normalize vocab
    activities_norm = [normalize(a) for a in activities]
    object_types_norm = [normalize(o) for o in object_types]

    result = {
        "event_instances": [],     # e.g. "event:E12"
        "object_instances": [],    # e.g. "purchase_order:PO_55"
        "object_types": [],        # e.g. "purchase_order"
        "activities": [],          # e.g. "approve purchase order"
        "constraints": []          # structured constraints
    }

    # Extract event instances: event:<eid>
    event_pattern = r"\bevent\s*:\s*([A-Za-z0-9_\-]+)\b"
    for eid in re.findall(event_pattern, q, flags=re.IGNORECASE):
        result["event_instances"].append(f"event:{eid}")

    # Extract object instances:  <object_type>:<oid>
    type_union = "|".join([re.escape(ot) for ot in object_types_norm])
    obj_inst_pattern = rf"\b({type_union})\s*:\s*([A-Za-z0-9_\-]+)\b"
    for obj_type, oid in re.findall(obj_inst_pattern, q_low):
        instance = f"{obj_type}:{oid}"
        result["object_instances"].append(instance)
        if obj_type not in result["object_types"]:
            result["object_types"].append(obj_type)

    # Extract object types mentioned without instances
    for ot in object_types_norm:
        if re.search(make_regex(ot), q_low):
            if ot not in result["object_types"]:
                result["object_types"].append(ot)

    # Extract activities
    for act in activities_norm:
        if re.search(make_regex(act), q_low):
            result["activities"].append(act)

    # Extract high-level constraints based on keywords
    if "before" in q_low:
        result["constraints"].append(("TEMP_BEFORE", result["activities"]))
    if "after" in q_low:
        result["constraints"].append(("TEMP_AFTER", result["activities"]))
    if "without" in q_low or "exclude" in q_low:
        result["constraints"].append(("NEG_OBJECT_TYPE", result["object_types"]))

    return result