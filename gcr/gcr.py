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
            timestamp=str(row["ocel:timestamp"]),
            objects=event_objects.get(eid, set()),
            object_types=event_object_types.get(eid, set()),
        )

    return events


def enrich_paths_with_context(
    paths: List[List["Event"]],
    anchor_object: str,
    G_context: nx.DiGraph,
) -> str:
    """
    Enrich a list of constrained event paths with object context fetched from
    G_context, producing a structured natural-language string for injection into
    a large LLM prompt.

    Because the paths are guaranteed valid graph walks (produced by the
    ProcessTrie), every object retrieved here is provably connected to the
    anchor — a stronger grounding guarantee than similarity-based retrieval.

    For each event in each path the function collects:
      - Objects that participated in that event (via participation edges)
      - Each object's stored attributes (all node attrs except bookkeeping keys)
      - Object-to-object relations involving those objects (o2o edges)

    Deduplication is applied at the object level across all paths so the same
    object description never appears twice in the context block.

    Parameters
    ----------
    paths:
        Output of enumerate_object_valid_paths — lists of Event objects.
    anchor_object:
        The query anchor (e.g. "material:835"). Always described first.
    G_context:
        Heterogeneous context graph (Event + Object nodes, participation /
        o2o edges) built by ocel_to_graph_with_pm4py.

    Returns
    -------
    str
        Structured context block ready to be inserted into an LLM prompt, e.g.:

            Anchor object: material:835 (type: material)

            Path 1: Create Purchase Requisition → Approve Purchase Requisition
              Step 1 — Create Purchase Requisition
                Objects involved: purchase_requisition:12, material:835
                  purchase_requisition:12 → related to: purchase_order:55 [o2o]
              Step 2 — Approve Purchase Requisition
                Objects involved: purchase_requisition:12

            [Object details]
              material:835 (type: material)
                Attributes: price=120.0, unit=EA
              purchase_requisition:12 (type: purchase_requisition)
                Attributes: quantity=5
    """
    _SKIP_ATTRS = {"entity_type", "object_type"}

    def _obj_type(oid: str) -> str:
        if oid not in G_context.nodes:
            return ""
        return G_context.nodes[oid].get("object_type", "")

    def _obj_attrs(oid: str) -> dict:
        if oid not in G_context.nodes:
            return {}
        return {
            k: v for k, v in G_context.nodes[oid].items()
            if k not in _SKIP_ATTRS
            and not k.startswith("ocel:")
            and str(v) not in ("", "nan", "None", "none")
        }

    def _o2o_relations(oid: str) -> List[Tuple[str, str]]:
        """[(neighbour_id, relation_label), ...] for object-to-object edges."""
        if oid not in G_context.nodes:
            return []
        rels = []
        for _, nbr, data in G_context.out_edges(oid, data=True):
            if G_context.nodes.get(nbr, {}).get("entity_type") == "Object":
                rels.append((nbr, data.get("label", data.get("edge_type", "related_to"))))
        for src, _, data in G_context.in_edges(oid, data=True):
            if G_context.nodes.get(src, {}).get("entity_type") == "Object":
                rels.append((src, data.get("label", data.get("edge_type", "related_to"))))
        return rels

    lines: List[str] = []
    seen_objects: Set[str] = set()
    object_detail_lines: List[str] = []

    lines.append(f"Anchor object: {anchor_object} (type: {_obj_type(anchor_object)})")
    lines.append("")

    for path_idx, path in enumerate(paths, start=1):
        lines.append(f"Path {path_idx}: {' → '.join(e.activity for e in path)}")

        for step_idx, event in enumerate(path, start=1):
            lines.append(f"  Step {step_idx} — {event.activity}")

            step_objs = sorted(event.objects)
            if step_objs:
                lines.append(f"    Objects involved: {', '.join(step_objs)}")
            for oid in step_objs:
                o2o = _o2o_relations(oid)
                if o2o:
                    rel_str = ", ".join(f"{nbr} [{lbl}]" for nbr, lbl in o2o)
                    lines.append(f"      {oid} → related to: {rel_str}")
                if oid not in seen_objects:
                    seen_objects.add(oid)
                    attrs = _obj_attrs(oid)
                    detail = f"  {oid} (type: {_obj_type(oid)})"
                    if attrs:
                        detail += "\n    Attributes: " + ", ".join(
                            f"{k}={v}" for k, v in attrs.items()
                        )
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