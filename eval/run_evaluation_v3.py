"""
eval/run_evaluation.py
=======================
Orchestrates the full contrastive evaluation pipeline over four conditions:

  1. constrained_gcr   — Trie-constrained path generation (small LLM) used as
                         context for answer generation (large LLM). This work.
  2. unconstrained_gcr — Same architecture, no Trie constraint. Ablation
                         baseline mirroring "GCR w/o constraint" in Luo et al.
                         (2024).
  3. graphrag          — Local subgraph context (1-hop BFS) + large LLM.
  4. rag               — Dense FAISS retrieval + large LLM.

The two GCR conditions use a two-stage pipeline: a small path-generation LLM
(paths pre-generated and supplied as JSONL files) feeds context into the same
large generation LLM used by graphrag and rag. This ensures the only variable
across conditions is the retrieval/context mechanism, not the generator.

Evaluation is two-stage, following the question typology of the dataset:

  Stage 1 — Path retrieval quality (GCR conditions only):
      activity_recall  : fraction of ground-truth Event:Activity_Name nodes
                         found in the retrieved paths.
      entity_recall    : fraction of ground-truth Object:type nodes found
                         (Type 3 cross-object questions only).

  Stage 2 — Answer generation quality (all conditions):
      Types 1 & 3 — LLM-as-judge faithfulness (1-5 scale) + hallucination flag.
                    Judge model: GPT-4o (independent of the model under test),
                    following Es et al. (2023).
      Type 2      — Binary precision / recall / F1 on the anomaly label +
                    deviation point accuracy. No LLM judge required: ground
                    truth is determined automatically by variant analysis.

Note on path-based metrics for RAG / GraphRAG:
    These baselines return free-text answers, not structured reasoning paths.
    Stage 1 metrics are therefore not computed for rag and graphrag — this is
    correct behaviour and should be reported as such in the paper.

Usage
-----
    python -m eval.run_evaluation \\
        --dataset      eval/data/eval_combined.jsonl \\
        --graphml      test2.graphml \\
        --faiss_db     ./faiss_db_minilm \\
        --model_id     Qwen/Qwen2.5-7B-Instruct \\
        --judge_model  gpt-4o \\
        --constrained_paths   predicted_paths_constrained.jsonl \\
        --unconstrained_paths predicted_paths_unconstrained.jsonl \\
        --conditions   rag,graphrag,unconstrained_gcr,constrained_gcr \\
        --out_dir      results
        
        python -m eval.run_evaluation_v3 --dataset eval/data/eval_v2_combined.jsonl --graphml test2.graphml --faiss_db ./faiss_db_minilm --model_id Qwen/Qwen2.5-7B-Instruct --judge_model gpt-4o --constrained_paths   results/predicted_paths_constrained.jsonl --unconstrained_paths results/predicted_paths_unconstrained.jsonl --conditions rag,graphrag,unconstrained_gcr,constrained_gcr --out_dir results


    For quick testing:
        --model_id Qwen/Qwen2.5-1.5B-Instruct --limit 5

Predicted paths format (one JSONL line per question):
    {"id": "T1_001", "paths": ["Event:Create_Purchase_Order NEXT_FOR_po ..."]}

Dependencies
------------
    pip install transformers torch openai langchain-community faiss-cpu
                networkx tqdm python-dotenv
    Requires OPENAI_API_KEY in environment or .env for the LLM judge.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx
import torch
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Resolve project root so imports work regardless of cwd
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

load_dotenv()
judge_client = OpenAI()


# =============================================================================
# GRAPH UTILITIES
# =============================================================================

def load_graph(graphml_path: str) -> nx.DiGraph:
    G = nx.read_graphml(graphml_path)
    print(f"[graph] Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def get_local_subgraph_text(
    G: nx.DiGraph,
    topic_entities: List[str],
    max_depth: int = 2,
) -> str:
    """
    GraphRAG condition: build a text description of the local subgraph around
    the topic entities up to *max_depth* hops.  The BFS is bounded to keep
    context length manageable for the generation LLM.
    """
    visited: set = set()
    lines: List[str] = []

    def traverse(node: str, depth: int) -> None:
        if node in visited or depth > max_depth:
            return
        visited.add(node)
        data = G.nodes[node]
        entity_type = data.get("entity_type", "")
        if entity_type == "Event":
            lines.append(
                f"Event {node} | Activity: {data.get('activity', '')} | "
                f"Timestamp: {data.get('timestamp', '')}"
            )
        elif entity_type == "Object":
            lines.append(f"Object {data.get('object_type', 'object')}:{node}")
        for _, nxt, edata in G.out_edges(node, data=True):
            lines.append(f"  --[{edata.get('label', 'rel')}]--> {nxt}")
            traverse(nxt, depth + 1)

    for ent in topic_entities:
        if ent in G:
            traverse(ent, 0)

    return "\n".join(lines) if lines else "No context found in graph."


# =============================================================================
# HF GENERATION MODEL
# =============================================================================

def load_hf_model(model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    print(f"[model] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    print(f"[model] Loaded on {next(model.parameters()).device}")
    return tokenizer, model


def call_hf_model(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    system: Optional[str] = None,
    max_new_tokens: int = 512,
) -> str:
    """
    Generate a response using the HF model's chat template.
    Compatible with Qwen2.5-Instruct and other instruction-tuned models.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# =============================================================================
# RAG RETRIEVAL
# =============================================================================

def get_rag_context(
    question: str,
    faiss_db_path: str,
    embedding_backend: str = "minilm",
    k: int = 5,
) -> str:
    """
    RAG condition: retrieve top-k documents from the FAISS index.
    Lazy-imports langchain to avoid hard dependency at module level.
    """
    from rag.rag import get_retriever_from_db
    retriever = get_retriever_from_db(
        faiss_db_path, embedding_backend=embedding_backend, k=k
    )
    docs = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in docs)


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

SYSTEM_PROCESS_MINING = (
    "You are a process mining assistant. Answer questions about business "
    "process event logs accurately and concisely, based only on the "
    "information provided."
)

SYSTEM_JUDGE = (
    "You are an expert evaluator for process mining question answering "
    "systems. Your task is to assess the faithfulness and correctness of "
    "an answer against a set of ground truth facts extracted from a "
    "process event log."
)


def build_context_prompt(question: str, context: str) -> str:
    return (
        f"Use the following process context to answer the question.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"Answer based strictly on the context provided."
    )


def build_noctx_prompt(question: str) -> str:
    return f"Answer the following question about a business process:\n\nQUESTION:\n{question}"


# =============================================================================
# STAGE 1 — PATH HIT METRIC  (GCR conditions only)
# =============================================================================

# def hit_metric_paths(
#     predicted_paths: List[str],
#     ground_truth: Dict,
# ) -> Dict:
#     """
#     Evaluate whether retrieved paths contain the ground-truth entities.
#     Expected nodes are formatted as Event:Activity_Name (underscored),
#     matching the linearize_path() output format in gcr/gcr.py.

#     Returns
#     -------
#     dict with keys:
#         activity_recall : float | None  — fraction of ground-truth Event nodes found
#         entity_recall   : float | None  — fraction of Object nodes found (Type 3)
#         hits / misses   : lists of matched/missed node strings
#     """
#     pgt = ground_truth.get("path_ground_truth", {})
#     expected_nodes = pgt.get("expected_nodes", [])
#     object_nodes = pgt.get("object_nodes", [])

#     if not expected_nodes and not object_nodes:
#         return {"activity_recall": None, "note": "no path_ground_truth in record"}

#     all_paths_text = " ".join(predicted_paths).lower()

#     if expected_nodes:
#         hits = [n for n in expected_nodes if n.lower() in all_paths_text]
#         activity_recall = round(len(hits) / len(expected_nodes), 3)
#         misses = [n for n in expected_nodes if n not in hits]
#     else:
#         activity_recall, hits, misses = None, [], []

#     result: Dict = {
#         "activity_recall": activity_recall,
#         "hits": hits,
#         "misses": misses,
#     }

#     if object_nodes:
#         obj_hits = [n for n in object_nodes if n.lower() in all_paths_text]
#         result["entity_recall"] = round(len(obj_hits) / len(object_nodes), 3)
#         result["object_hits"] = obj_hits

#     return result
def hit_metric_paths(
        predicted_paths: List[str],
        ground_truth: Dict,
    ) -> Dict:
        """
        Evaluate whether retrieved GCR paths contain the ground-truth entities.

        Expected nodes are formatted as "Event:Activity_Name" or "Object:type"
        (from path_ground_truth in the eval dataset). GCR paths are chain strings
        like "Event:Create_PO NEXT_FOR_po Event:Approve_PO ...".

        Two normalisation steps are applied before matching:
        1. Deduplication — expected_nodes often contains repeated entries (one
            per object event sequence), which would unfairly inflate the
            denominator. We evaluate against the unique set.
        2. Object label normalisation — expected_nodes uses "Object:goods receipt"
            (space) but GCR relation labels use "Object:goods_receipt" (underscore).
            Both forms are checked.

        Returns
        -------
        dict with keys:
            activity_recall : float | None  — fraction of unique ground-truth
                            Event nodes found in the predicted paths
            entity_recall   : float | None  — fraction of unique Object nodes found
                            (Type 3 cross-object questions only)
            hits / misses   : lists of matched/missed node label strings
        """
        pgt = ground_truth.get("path_ground_truth", {})
        expected_nodes = pgt.get("expected_nodes", [])
        object_nodes = pgt.get("object_nodes", [])

        if not expected_nodes and not object_nodes:
            return {"activity_recall": None, "note": "no path_ground_truth in record"}

        # Join all predicted path strings into one lowercase search space
        all_paths_text = " ".join(predicted_paths).lower()

        # --- Activity recall (Event nodes) ---
        # Deduplicate: a node that appears 4x in expected_nodes should only count once.
        unique_event_nodes = list(dict.fromkeys(expected_nodes))  # preserves order

        if unique_event_nodes:
            hits, misses = [], []
            for n in unique_event_nodes:
                # Normalise spaces to underscores so "Event:Two-Way Match" matches
                # "Event:Two-Way_Match" in GCR output
                n_normalised = n.lower().replace(" ", "_")
                if n_normalised in all_paths_text or n.lower() in all_paths_text:
                    hits.append(n)
                else:
                    misses.append(n)
            activity_recall = round(len(hits) / len(unique_event_nodes), 3)
        else:
            activity_recall, hits, misses = None, [], []

        result: Dict = {
            "activity_recall": activity_recall,
            "hits": hits,
            "misses": misses,
            "unique_expected": len(unique_event_nodes),
            "raw_expected": len(expected_nodes),  # for transparency in results
        }

        # --- Entity recall (Object nodes, Type 3 only) ---
        if object_nodes:
            unique_object_nodes = list(dict.fromkeys(object_nodes))
            obj_hits = []
            for n in unique_object_nodes:
                # "Object:goods receipt" → also try "Object:goods_receipt"
                n_lower = n.lower()
                n_underscored = n_lower.replace(" ", "_")
                if n_lower in all_paths_text or n_underscored in all_paths_text:
                    obj_hits.append(n)
            result["entity_recall"] = round(len(obj_hits) / len(unique_object_nodes), 3)
            result["object_hits"] = obj_hits
            result["unique_object_expected"] = len(unique_object_nodes)

        return result

# =============================================================================
# STAGE 2 — ANSWER QUALITY METRICS
# =============================================================================

def judge_faithfulness(
    question: str,
    answer: str,
    ground_truth_facts: Dict,
    judge_model: str,
) -> Dict:
    """
    LLM-as-judge faithfulness scoring (1-5 scale) with hallucination flag.

    The judge model (default: GPT-4o) is independent of the model under test,
    following the evaluation protocol of Es et al. (2023).  The judge is
    prompted with the ground-truth facts extracted from the graph, not with
    the raw graph itself, to keep the evaluation condition controlled.

    Returns
    -------
    dict with keys: score (int), reasoning (str), hallucination_flag (bool)
    """
    facts_str = json.dumps(ground_truth_facts, indent=2)
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER TO EVALUATE:\n{answer}\n\n"
        f"GROUND TRUTH FACTS (from process event log):\n{facts_str}\n\n"
        f"Evaluate the answer on the following criteria:\n"
        f"1. FAITHFULNESS (1-5): Does the answer correctly reflect the ground "
        f"truth facts? 1=completely wrong, 3=partially correct, 5=fully correct "
        f"and faithful.\n"
        f"2. HALLUCINATION: Does the answer mention activities, objects, or "
        f"events NOT present in the ground truth facts? (yes/no)\n\n"
        f"Respond in this exact JSON format:\n"
        f'{{"score": <1-5>, "reasoning": "<brief explanation>", '
        f'"hallucination": "<yes/no>"}}'
    )

    response = judge_client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": SYSTEM_JUDGE},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()

    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(clean)
        return {
            "score": int(parsed.get("score", 0)),
            "reasoning": parsed.get("reasoning", ""),
            "hallucination_flag": parsed.get("hallucination", "no").lower() == "yes",
        }
    except Exception:
        return {
            "score": 0,
            "reasoning": f"Parse error: {raw}",
            "hallucination_flag": False,
        }


def evaluate_anomaly_detection(answer: str, ground_truth: Dict) -> Dict:
    """
    Binary evaluation for Type 2 anomaly detection questions.

    The majority variant is explicitly stated in every question template, so
    the model's answer is tested on comparison-reasoning rather than domain
    knowledge.  No LLM judge is needed: classification is performed by
    keyword matching against a fixed set of conformance / non-conformance terms.

    Returns
    -------
    dict with keys:
        predicted_anomalous     : bool
        actual_anomalous        : bool
        correct                 : bool
        deviation_point_correct : bool | None (None when ground truth is normal)
    """
    is_anomalous = ground_truth.get("label", False)
    deviation = ground_truth.get("deviation", "")
    answer_lower = answer.lower()

    anomaly_keywords = [
        "does not conform", "not conform", "missing", "skipped",
        "out of order", "deviation", "does not match", "not match",
        "absent", "omitted", "incorrect order", "wrong order",
        "unexpected", "anomal", "non-conforman", "not standard",
        "differs from", "deviates",
    ]
    normal_keywords = [
        "conforms", "conform to", "follows the standard", "matches the standard",
        "standard flow", "correct order", "nothing unusual", "no deviation",
        "no missing", "consistent with", "aligns with", "as expected",
    ]

    predicted_anomalous = any(kw in answer_lower for kw in anomaly_keywords)
    predicted_normal = any(kw in answer_lower for kw in normal_keywords)

    # Resolve ambiguity: anomaly keywords take precedence if both fire
    if predicted_anomalous and not predicted_normal:
        predicted = True
    elif predicted_normal and not predicted_anomalous:
        predicted = False
    else:
        predicted = predicted_anomalous

    deviation_correct: Optional[bool] = None
    if is_anomalous and deviation:
        deviation_terms = [
            t.strip("[]'\"").lower()
            for t in re.split(r"[,;:\s]+", deviation)
            if len(t) > 3
        ]
        deviation_correct = any(term in answer_lower for term in deviation_terms)

    return {
        "predicted_anomalous": predicted,
        "actual_anomalous": is_anomalous,
        "correct": predicted == is_anomalous,
        "deviation_point_correct": deviation_correct,
    }


# =============================================================================
# PER-CONDITION RUNNER
# =============================================================================

def run_condition(
    questions: List[Dict],
    condition: str,
    G: nx.DiGraph,
    tokenizer: AutoTokenizer,
    hf_model: AutoModelForCausalLM,
    judge_model: str,
    out_path: str,
    faiss_db_path: Optional[str] = None,
    embedding_backend: str = "minilm",
    predicted_paths_map: Optional[Dict[str, List[str]]] = None,
) -> List[Dict]:
    """
    Run evaluation for one condition over all questions and write per-item JSONL.

    Parameters
    ----------
    condition:
        One of: "rag" | "graphrag" | "unconstrained_gcr" | "constrained_gcr"
    predicted_paths_map:
        Dict of {question_id -> [path strings]} pre-generated by the small LLM.
        Required for the two GCR conditions.
    """
    # Validate required inputs per condition
    if condition == "rag" and not faiss_db_path:
        raise ValueError("--faiss_db is required for the rag condition.")
    if condition in ("unconstrained_gcr", "constrained_gcr") and not predicted_paths_map:
        flag = "constrained" if condition == "constrained_gcr" else "unconstrained"
        raise ValueError(
            f"--{flag}_paths is required for the {condition} condition."
        )

    results: List[Dict] = []
    print(f"\n[{condition.upper()}] Evaluating {len(questions)} questions...")

    for q in tqdm(questions):
        question_text = q["question"]
        topic_entities = q["topic_entities"]
        q_type = q["type"]
        gt = q["ground_truth"]

        t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # Build context and generate answer
        # ------------------------------------------------------------------
        if condition == "rag":
            context = get_rag_context(
                question_text, faiss_db_path, embedding_backend=embedding_backend
            )
            prompt = build_context_prompt(question_text, context)
            answer = call_hf_model(prompt, tokenizer, hf_model, SYSTEM_PROCESS_MINING)

        elif condition == "graphrag":
            context = get_local_subgraph_text(G, topic_entities)
            prompt = build_context_prompt(question_text, context)
            answer = call_hf_model(prompt, tokenizer, hf_model, SYSTEM_PROCESS_MINING)

        elif condition in ("unconstrained_gcr", "constrained_gcr"):
            predicted_paths = predicted_paths_map.get(q["id"], [])
            if predicted_paths:
                context = "\n".join(predicted_paths)
                prompt = build_context_prompt(question_text, context)
            else:
                # Graceful fallback: no paths available for this question
                prompt = build_noctx_prompt(question_text)
            #answer = call_hf_model(prompt, tokenizer, hf_model, SYSTEM_PROCESS_MINING)
            answer = ""
        else:
            raise ValueError(f"Unknown condition: '{condition}'")

        elapsed = time.perf_counter() - t0

        # ------------------------------------------------------------------
        # Build result record
        # ------------------------------------------------------------------
        result: Dict = {
            "id": q["id"],
            "type": q_type,
            "condition": condition,
            "question": question_text,
            "answer": answer,
            "generation_s": round(elapsed, 3),
        }

        # Stage 1 — path hit metric (GCR conditions only)
        if condition in ("unconstrained_gcr", "constrained_gcr"):
            predicted_paths = predicted_paths_map.get(q["id"], [])
            result["path_hit_metric"] = hit_metric_paths(predicted_paths, gt)

        # Stage 2 — answer quality
        if q_type == "anomaly_detection":
            # Binary classification: no LLM judge needed
            result["binary_eval"] = evaluate_anomaly_detection(answer, gt)
        # LLM-as-judge faithfulness for all question types
        # result["faithfulness"] = judge_faithfulness(
        #     question_text, answer, gt.get("facts", {}), judge_model
        # )

        results.append(result)

    # Write per-condition JSONL
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved {len(results)} records → {out_path}")

    return results


# =============================================================================
# SUMMARY AGGREGATION & REPORTING
# =============================================================================

def compute_summary(all_results: Dict[str, List[Dict]]) -> Dict:
    """
    Aggregate per-item results into condition-level summary statistics.

    Produces three metric groups:
      - faithfulness / hallucination_rate (all conditions, Types 1 & 3)
      - anomaly_detection: precision / recall / F1 / deviation_accuracy (Type 2)
      - path_hit_metric: activity_recall / entity_recall (GCR conditions only)
    """
    summary: Dict = {}

    for condition, results in all_results.items():
        cond: Dict = {"total": len(results)}

        # --- Faithfulness / hallucination (Types 1 & 3) ---
        faith_scores = [
            r["faithfulness"]["score"]
            for r in results
            if "faithfulness" in r and r["faithfulness"]["score"] > 0
        ]
        hallucination_flags = [
            r["faithfulness"]["hallucination_flag"]
            for r in results
            if "faithfulness" in r
        ]
        cond["mean_faithfulness"] = (
            round(sum(faith_scores) / len(faith_scores), 3) if faith_scores else None
        )
        cond["hallucination_rate"] = (
            round(sum(hallucination_flags) / len(hallucination_flags), 3)
            if hallucination_flags else None
        )

        # --- Anomaly detection (Type 2) ---
        t2 = [r for r in results if r["type"] == "anomaly_detection"]
        if t2:
            tp = sum(1 for r in t2 if r["binary_eval"]["predicted_anomalous"]
                     and r["binary_eval"]["actual_anomalous"])
            fp = sum(1 for r in t2 if r["binary_eval"]["predicted_anomalous"]
                     and not r["binary_eval"]["actual_anomalous"])
            fn = sum(1 for r in t2 if not r["binary_eval"]["predicted_anomalous"]
                     and r["binary_eval"]["actual_anomalous"])
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )
            dev_correct = [
                r["binary_eval"]["deviation_point_correct"]
                for r in t2
                if r["binary_eval"]["deviation_point_correct"] is not None
            ]
            cond["anomaly_detection"] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "deviation_accuracy": (
                    round(sum(dev_correct) / len(dev_correct), 3)
                    if dev_correct else None
                ),
            }

        # --- Path hit metric (GCR conditions only) ---
        path_hits = [r["path_hit_metric"] for r in results if "path_hit_metric" in r]
        if path_hits:
            act_recalls = [
                h["activity_recall"] for h in path_hits
                if h.get("activity_recall") is not None
            ]
            ent_recalls = [
                h["entity_recall"] for h in path_hits
                if h.get("entity_recall") is not None
            ]
            cond["path_hit_metric"] = {
                "mean_activity_recall": (
                    round(sum(act_recalls) / len(act_recalls), 3)
                    if act_recalls else None
                ),
                "mean_entity_recall": (
                    round(sum(ent_recalls) / len(ent_recalls), 3)
                    if ent_recalls else None
                ),
            }

        summary[condition] = cond

    return summary


def print_summary_table(summary: Dict) -> None:
    """Print a human-readable summary table to stdout."""
    print("\n========== EVALUATION SUMMARY ==========")

    # Header
    col_w = 20
    metric_cols = [
        ("mean_faithfulness",  "Faithfulness"),
        ("hallucination_rate", "Halluc.Rate"),
    ]
    header = f"{'Condition':<25}" + "".join(f"{name:>{col_w}}" for _, name in metric_cols)
    header += f"{'Anomaly-F1':>{col_w}}{'Act.Recall':>{col_w}}{'Ent.Recall':>{col_w}}"
    print(header)
    print("-" * len(header))

    for condition, cond in summary.items():
        row = f"{condition:<25}"
        for key, _ in metric_cols:
            val = cond.get(key)
            row += f"{(f'{val:.3f}' if val is not None else 'N/A'):>{col_w}}"
        anomaly_f1 = cond.get("anomaly_detection", {}).get("f1")
        row += f"{(f'{anomaly_f1:.3f}' if anomaly_f1 is not None else 'N/A'):>{col_w}}"
        path = cond.get("path_hit_metric", {})
        act_r = path.get("mean_activity_recall")
        ent_r = path.get("mean_entity_recall")
        row += f"{(f'{act_r:.3f}' if act_r is not None else 'N/A'):>{col_w}}"
        row += f"{(f'{ent_r:.3f}' if ent_r is not None else 'N/A'):>{col_w}}"
        print(row)

    print("=========================================\n")


# =============================================================================
# DATASET LOADER
# =============================================================================

def load_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_paths_file(path: Optional[str]) -> Optional[Dict[str, List[str]]]:
    """Load a predicted-paths JSONL file into a {question_id -> [paths]} dict."""
    if not path or not os.path.exists(path):
        return None
    result: Dict[str, List[str]] = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            result[rec["id"]] = rec.get("paths", [])
    print(f"Loaded predicted paths from {path} ({len(result)} entries).")
    return result


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GCR-OCEL contrastive evaluation runner."
    )
    p.add_argument(
        "--dataset", required=True,
        help="Path to the combined evaluation JSONL (e.g. eval/data/eval_combined.jsonl)"
    )
    p.add_argument(
        "--graphml", required=True,
        help="Path to the OCEL process graph (e.g. test2.graphml)"
    )
    p.add_argument(
        "--model_id", default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID used for answer generation across all conditions"
    )
    p.add_argument(
        "--judge_model", default="gpt-4o",
        help="OpenAI model used as independent LLM judge (Es et al., 2023)"
    )
    p.add_argument(
        "--faiss_db", default="./faiss_db_pm4py",
        help="Path to the saved FAISS index (required for the rag condition)"
    )
    p.add_argument(
        "--embedding", default="minilm",
        choices=["openai", "bge", "minilm", "e5"],
        help="Embedding backend used by the RAG retriever"
    )
    p.add_argument(
        "--constrained_paths", default=None,
        help=(
            "JSONL of Trie-constrained paths from the small path-generation LLM. "
            "One line per question: {\"id\": \"T1_001\", \"paths\": [...]}. "
            "Required for the constrained_gcr condition."
        )
    )
    p.add_argument(
        "--unconstrained_paths", default=None,
        help=(
            "JSONL of unconstrained paths from the small path-generation LLM. "
            "Same format as --constrained_paths. "
            "Required for the unconstrained_gcr condition."
        )
    )
    p.add_argument(
        "--conditions",
        default="rag,graphrag,unconstrained_gcr,constrained_gcr",
        help="Comma-separated list of conditions to run"
    )
    p.add_argument(
        "--out_dir", default="results",
        help="Directory where per-condition JSONL and summary JSON are written"
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of questions evaluated (useful for quick testing)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load graph and dataset
    # ------------------------------------------------------------------
    G = load_graph(args.graphml)

    questions = load_jsonl(args.dataset)
    if args.limit:
        questions = questions[:args.limit]
    print(f"Loaded {len(questions)} questions from {args.dataset}.")

    # Question type breakdown
    type_counts = Counter(q["type"] for q in questions)
    for t, c in type_counts.items():
        print(f"  {t}: {c}")

    # ------------------------------------------------------------------
    # Load predicted paths (GCR conditions)
    # ------------------------------------------------------------------
    constrained_paths_map = load_paths_file(args.constrained_paths)
    unconstrained_paths_map = load_paths_file(args.unconstrained_paths)

    # ------------------------------------------------------------------
    # Load HF model once — shared across all conditions
    # ------------------------------------------------------------------
    tokenizer, hf_model = load_hf_model(args.model_id)

    # ------------------------------------------------------------------
    # Run per-condition evaluation
    # ------------------------------------------------------------------
    conditions = [c.strip() for c in args.conditions.split(",")]
    all_results: Dict[str, List[Dict]] = {}

    for condition in conditions:
        paths_map = (
            constrained_paths_map if condition == "constrained_gcr"
            else unconstrained_paths_map if condition == "unconstrained_gcr"
            else None
        )
        out_path = os.path.join(args.out_dir, f"eval_results_{condition}.jsonl")

        try:
            results = run_condition(
                questions=questions,
                condition=condition,
                G=G,
                tokenizer=tokenizer,
                hf_model=hf_model,
                judge_model=args.judge_model,
                out_path=out_path,
                faiss_db_path=args.faiss_db,
                embedding_backend=args.embedding,
                predicted_paths_map=paths_map,
            )
            all_results[condition] = results
        except ValueError as e:
            print(f"  [WARNING] Skipping condition '{condition}': {e}")

    # ------------------------------------------------------------------
    # Aggregate and write summary
    # ------------------------------------------------------------------
    summary = compute_summary(all_results)
    print_summary_table(summary)

    summary_path = os.path.join(args.out_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Written: {summary_path}")

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
