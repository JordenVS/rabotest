from gcr.trie import ProcessTrie
from typing import List, Set, Dict, Tuple
from transformers import AutoTokenizer
import networkx as nx
from collections import defaultdict
from gcr.event import Event

def linearize_event_path(path: List[Event], sep: str = " ") -> str:
    """
    Convert an event-only path into a token-friendly string
    suitable for Trie construction and LLM context.
    """
    return sep.join(
        f"Event:{e.activity.replace(' ', '_')}"
        for e in path
    )

def build_event_successors_from_g_behavior(
    G_behavior: nx.DiGraph,
    events: Dict[str, Event],
) -> Dict[str, List[Event]]:

    event_successors = defaultdict(list)

    for u, v, data in G_behavior.edges(data=True):
        if data.get("edge_type") == "behavior":
            event_successors[u].append(events[v])

    return dict(event_successors)

def enumerate_object_valid_paths(
    event_successors: Dict[str, List[Event]],
    start_events: List[Event],
    anchor_object: str,
    max_depth: int = 5,
) -> List[List[Event]]:
    """
    Enumerate UNIQUE event-only paths that are valid under object-centric rules.

    Paths are considered identical if they share the same sequence of event IDs,
    regardless of object scope or traversal history.
    """

    all_paths: List[List[Event]] = []
    seen_paths: Set[Tuple[str, ...]] = set()

    # Stack entries: (current_path, active_objects, last_event)
    stack = []

    # Seed paths
    for e in start_events:
        if anchor_object in e.objects:
            stack.append(([e], set(e.objects), e))

    while stack:
        path, active_objects, last_event = stack.pop()

        # --- uniqueness check (event-ID level) ---
        path_key = tuple(evt.eid for evt in path)
        if path_key not in seen_paths:
            seen_paths.add(path_key)
            all_paths.append(path)

        if len(path) >= max_depth:
            continue

        # Expand path
        for e_next in event_successors.get(last_event.eid, []):
            # Object-centric validity constraint
            if e_next.objects & active_objects:
                stack.append(
                    (
                        path + [e_next],
                        active_objects | e_next.objects,
                        e_next,
                    )
                )

    return all_paths

def build_trie_from_path_strings(path_strings, tokenizer):

    if isinstance(tokenizer, str):
        tok = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    else:
        tok = tokenizer

    trie = ProcessTrie()
    for text in path_strings:
        ids = tok.encode(text, add_special_tokens=False)
        if ids:
            trie.insert(ids)
    return trie

def build_event_object_maps(ocel):
    """
    Builds:
      event_objects[eid] = set(object_ids)
      event_object_types[eid] = set(object_types)
    """
    event_objects = defaultdict(set)
    event_object_types = defaultdict(set)

    for _, row in ocel.relations.iterrows():
        eid = row["ocel:eid"]
        oid = row["ocel:oid"]
        otype = row["ocel:type"]

        event_objects[eid].add(oid)
        event_object_types[eid].add(otype)

    return event_objects, event_object_types

def build_events_dict(ocel) -> Dict[str, Event]:
    events_df = ocel.events

    event_objects, event_object_types = build_event_object_maps(ocel)

    events = {}

    for _, row in events_df.iterrows():
        eid = row["ocel:eid"]
        activity = row["ocel:activity"]

        events[eid] = Event(
            eid=eid,
            activity=activity,
            objects=event_objects.get(eid, set()),
            object_types=event_object_types.get(eid, set()),
        )

    return events

def reify_generated_path(self, generated_string, anchor_object, G_context):
    # 1. Robust Normalization (Lowercasing is key)
    activity_names = [
        s.replace("Event:", "").replace("_", " ").strip().lower() 
        for s in generated_string.split()
    ]
    
    if anchor_object not in G_context:
        return [None] * len(activity_names)

    # 2. Track "Active" Objects (Start with anchor, expand as we find events)
    active_objects = {anchor_object}
    reified_path = []
    used_eids = set()

    for act_name in activity_names:
        matched_event = None
        
        # Search neighbors of ALL currently active objects
        for obj in active_objects:
            candidate_neighbors = G_context.neighbors(obj)
            
            # Sort neighbors by timestamp to stay process-aware
            candidates = []
            for nbr in candidate_neighbors:
                node = G_context.nodes[nbr]
                if node.get("entity_type") == "Event" and nbr not in used_eids:
                    candidates.append((nbr, node))
            
            # Sort by timestamp (handling None safely)
            candidates.sort(key=lambda x: str(x[1].get("timestamp") or "0000"))

            for eid, node in candidates:
                # Case-insensitive matching
                if node.get("activity", "").lower() == act_name:
                    matched_event = self.events.get(eid)
                    used_eids.add(eid)
                    
                    # OBJECT HOP: Add all objects involved in THIS event to active_objects
                    # This allows the next activity to be found via a related object
                    if matched_event:
                        active_objects.update(matched_event.objects)
                    break
            
            if matched_event:
                break
        
        reified_path.append(matched_event)
        if not matched_event:
             print(f" [DEBUG] Failed to match: {act_name} among neighbors of {active_objects}")
                
    return reified_path

def enrich_paths_with_context(
    paths: List[List["Event"]],
    anchor_object: str,
    G_context: nx.DiGraph,
    max_depth: int = 5,
) -> str:
    """
    Enriches paths with focused context. 
    - Inline relations are restricted to objects present in the same step.
    - Broad neighbor context and attributes are moved to a single 'Object details' footer.
    """
    _SKIP_ATTRS = {"entity_type", "object_type"}
    MAX_NEIGHBORS = 5  # Cap to prevent token bloat

    def _obj_type(oid: str) -> str:
        if oid not in G_context.nodes:
            return "unknown"
        return G_context.nodes[oid].get("object_type", "object")

    def _obj_attrs(oid: str) -> dict:
        if oid not in G_context.nodes:
            return {}
        return {
            k.replace("ocel:", ""): v for k, v in G_context.nodes[oid].items()
            if k not in _SKIP_ATTRS
            and not k.startswith("ocel:")
            and str(v).lower() not in ("", "nan", "none")
        }

    def _get_focused_relations(oid: str, step_objs: Set[str]) -> List[Tuple[str, str]]:
        """Only returns relations to other objects involved in the CURRENT step."""
        if oid not in G_context.nodes:
            return []
        rels = []
        seen = set()
        # Check edges to see if any neighbor is also in this step
        edges = list(G_context.out_edges(oid, data=True)) + list(G_context.in_edges(oid, data=True))
        for u, v, data in edges:
            nbr = v if u == oid else u
            if nbr in step_objs and nbr != oid:
                label = data.get("label", data.get("edge_type", "related_to"))
                if (nbr, label) not in seen:
                    rels.append((nbr, label))
                    seen.add((nbr, label))
        return rels

    def _get_all_neighbors(oid: str, limit: int = 5) -> List[Tuple[str, str]]:
        """Returns a capped list of all unique object neighbors for the footer."""
        if oid not in G_context.nodes:
            return []
        rels = []
        seen = set()
        edges = list(G_context.out_edges(oid, data=True)) + list(G_context.in_edges(oid, data=True))
        for u, v, data in edges:
            nbr = v if u == oid else u
            if G_context.nodes.get(nbr, {}).get("entity_type") == "Object" and nbr != oid:
                label = data.get("label", data.get("edge_type", "related_to"))
                if (nbr, label) not in seen:
                    rels.append((nbr, label))
                    seen.add((nbr, label))
            if len(rels) >= limit:
                break
        return rels

    lines: List[str] = []
    seen_objects: Set[str] = set()
    object_detail_lines: List[str] = []

    lines.append(f"Anchor object: {anchor_object} (type: {_obj_type(anchor_object)})")
    lines.append("")

    for path_idx, path in enumerate(paths, start=1):
        lines.append(f"Path {path_idx}: {' → '.join(e.activity for e in path if e)}")

        for step_idx, event in enumerate(path, start=1):
            if not event: continue
            lines.append(f"  Step {step_idx} — {event.activity}")

            step_objs = set(event.objects)
            if step_objs:
                lines.append(f"    Objects involved: {', '.join(sorted(step_objs))}")
            
            for oid in step_objs:
                # 1. Inline: Only show relations to other objects in this specific event
                step_rels = _get_focused_relations(oid, step_objs)
                if step_rels:
                    rel_str = ", ".join(f"{nbr} [{lbl}]" for nbr, lbl in step_rels)
                    lines.append(f"      {oid} → connected to: {rel_str}")
                
                # 2. Detail Collection: Only add to footer once
                if oid not in seen_objects:
                    seen_objects.add(oid)
                    attrs = _obj_attrs(oid)
                    all_nbrs = _get_all_neighbors(oid, limit=max_depth)
                    
                    detail = f"  {oid} (type: {_obj_type(oid)})"
                    if attrs:
                        detail += "\n    Attributes: " + ", ".join(f"{k}={v}" for k, v in attrs.items())
                    if all_nbrs:
                        nbr_str = ", ".join(f"{nbr} [{lbl}]" for nbr, lbl in all_nbrs)
                        detail += f"\n    General Relations: {nbr_str}"
                    object_detail_lines.append(detail)

        lines.append("")

    if object_detail_lines:
        lines.append("[Object details]")
        lines.extend(object_detail_lines)

    return "\n".join(lines)

def build_events_dict_from_context_graph(G_context: nx.Graph) -> Dict[str, Event]:
    """
    Reconstruct event objects from a saved OCEL context graph.

    This avoids reloading the original OCEL dataset by inferring event-object
    memberships from participation edges in the graph.
    """
    event_objects = defaultdict(set)
    event_object_types = defaultdict(set)

    for u, v, data in G_context.edges(data=True):
        if data.get("edge_type") != "participation":
            continue

        u_type = G_context.nodes[u].get("entity_type")
        v_type = G_context.nodes[v].get("entity_type")

        if u_type == "Event" and v_type == "Object":
            event_objects[u].add(v)
            event_object_types[u].add(
                G_context.nodes[v].get("object_type", "")
            )
        elif u_type == "Object" and v_type == "Event":
            event_objects[v].add(u)
            event_object_types[v].add(
                G_context.nodes[u].get("object_type", "")
            )

    events = {}
    for node_id, attrs in G_context.nodes(data=True):
        if attrs.get("entity_type") != "Event":
            continue

        events[node_id] = Event(
            eid=node_id,
            activity=attrs.get("activity", ""),
            objects=event_objects.get(node_id, set()),
            object_types=event_object_types.get(node_id, set()),
        )

    return events