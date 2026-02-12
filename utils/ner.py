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
        "event_instances": [],     # e.g. "event:12"
        "object_instances": [],    # e.g. "purchase_order:55"
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

def normalize(term: str) -> str:
    """Normalize object/activity names to lowercase space-separated form."""
    return term.lower().replace("_", " ").strip()

def normalize_underscored(term: str) -> str:
    """Normalize to lowercase underscore-separated form."""
    return term.lower().replace(" ", "_").strip()

def make_regex(term: str) -> str:
    """Boundary-safe regex for multi-word matching."""
    t = re.escape(term)
    return rf"\b{t}\b"

def extract_entities_ocel(
    query: str,
    activities: List[str],
    object_types: List[str]
) -> Dict[str, Any]:
    """
    Extract:
      - event:<eid>
      - <object_type>:<oid>
      - standalone <object_type>
      - standalone activities
      - simple constraints
    Fully fixes:
      (1) purchase_order:20 not recognized
      (2) quotation:0 causing type expansion
    """

    q = query
    q_low = q.lower()

    # ------- Normalize vocab -------
    activities_norm = [normalize(a) for a in activities]  
    object_types_norm = [normalize(o) for o in object_types]        # e.g. "purchase order"
    object_types_underscored = [normalize_underscored(o) for o in object_types]  # e.g. "purchase_order"

    result = {
        "event_instances": [],
        "object_instances": [],
        "object_types": [],
        "activities": [],
        "constraints": []
    }

    # =====================================================
    # 1. EVENT INSTANCES (event:<eid>)
    # =====================================================
    event_pattern = r"\bevent\s*:\s*([A-Za-z0-9_\-]+)\b"
    for eid in re.findall(event_pattern, q_low):
        result["event_instances"].append(f"event:{eid}")

    # =====================================================
    # 2. OBJECT INSTANCES (<type>:<oid>)
    #    Must match BOTH "purchase_order" AND "purchase order"
    # =====================================================
    type_alternatives = set()

    # Add underscored and spaced versions:
    for t_us in object_types_underscored:
        type_alternatives.add(t_us)                # purchase_order
        type_alternatives.add(t_us.replace("_", " "))  # purchase order

    type_union = "|".join([re.escape(t) for t in type_alternatives])

    obj_inst_pattern = rf"\b({type_union})\s*:\s*([A-Za-z0-9_\-]+)\b"
    for otype_raw, oid in re.findall(obj_inst_pattern, q_low):
        canonical = normalize_underscored(otype_raw)  # ensures "purchase order" → purchase_order
        result["object_instances"].append(f"{canonical}:{oid}")

        # object_instances DOES NOT imply standalone object type
        if canonical not in result["object_types"]:
            result["object_types"].append(canonical)

    # =====================================================
    # 3. STANDALONE OBJECT TYPES
    #    Must NOT fire when mention is "<type>:<id>"
    # =====================================================
    for ot in object_types_norm:
        # only match if NOT followed by ":" (prevents "quotation:0" triggering)
        standalone_pat = rf"\b{re.escape(ot)}\b(?!\s*:)"
        if re.search(standalone_pat, q_low):
            canonical = normalize_underscored(ot)
            if canonical not in result["object_types"]:
                result["object_types"].append(canonical)

    # =====================================================
    # 4. ACTIVITIES (unchanged)
    # =====================================================
    for act in activities_norm:
        if re.search(make_regex(act), q_low):
            result["activities"].append(act)

    # =====================================================
    # 5. SIMPLE CONSTRAINTS
    # =====================================================
    if "before" in q_low:
        result["constraints"].append(("TEMP_BEFORE", result["activities"]))
    if "after" in q_low:
        result["constraints"].append(("TEMP_AFTER", result["activities"]))
    if "without" in q_low or "exclude" in q_low:
        result["constraints"].append(("NEG_OBJECT_TYPE", result["object_types"]))

    return result