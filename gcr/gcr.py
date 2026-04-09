from event import Event
from trie import ProcessTrie
from typing import List, Set, Dict, Tuple
from transformers import AutoTokenizer
import networkx as nx
from collections import defaultdict

def linearize_event_path(path: List[Event], sep: str = " ") -> str:
    """
    Convert an event-only path into a token-friendly string
    suitable for Trie construction and LLM context.
    """
    return sep.join(
        f"Event:{e.activity.replace(' ', '_')}"
        for e in path
    )

def build_event_successors_from_g_behavior(G_behavior: nx.DiGraph):
    """
    Build EVENT_SUCCESSORS from the behavioral graph.

    Returns:
      dict: event_id -> list of successor event_ids
    """
    event_successors = defaultdict(list)

    for u, v, data in G_behavior.edges(data=True):
        # Only follow behavioral edges
        if data.get("edge_type") == "behavior":
            event_successors[u].append(v)

    return dict(event_successors)
    
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