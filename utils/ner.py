import re

def extract_entities(query, object_types, activities):
    entities = {
        "object_instances": [],
        "object_types": [],
        "activities": [],
        "constraints": []
    }

    q = query.lower()

    # Object instances like order:O123
    for obj_type, obj_id in re.findall(r"\b([a-zA-Z_]+):([A-Z0-9]+)\b", query):
        entities["object_instances"].append(f"{obj_type}:{obj_id}")
        entities["object_types"].append(obj_type.lower())

    # Object types mentioned without ID
    for ot in object_types:
        if ot in q and ot not in entities["object_types"]:
            entities["object_types"].append(ot)

    # Activities (dictionary-based)
    for act in activities:
        if act in q:
            entities["activities"].append(act)

    # Constraints
    if "without" in q:
        entities["constraints"].append(("NEG_OBJECT_TYPE", entities["object_types"]))

    if "before" in q:
        entities["constraints"].append(("TEMPORAL_BEFORE", entities["activities"]))

    return entities
