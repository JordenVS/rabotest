"""
run_evaluation.py
==================
Runs a four-condition contrastive evaluation over the generated dataset.

Conditions
----------
1. rag              — Standard RAG: FAISS retrieval + HF model
2. graphrag         — GraphRAG: local subgraph context + HF model
3. unconstrained_gcr — Small HF LLM generates paths freely (no Trie) + large HF model
4. constrained_gcr  — Small HF LLM generates paths with Trie constraint + large HF model
                      (the method under test)

All four conditions use the same large HF model for answer generation so that
the only variable between conditions is the retrieval/context mechanism.
The LLM judge uses GPT-4o as an independent scoring tool, following standard
RAG evaluation practice (Es et al., 2023).

Metrics
-------
Stage 1 — Path retrieval quality (constrained_gcr and unconstrained_gcr only):
    activity_recall:  fraction of ground truth Event:Activity_Name nodes found
                      in the retrieved paths
    entity_recall:    fraction of ground truth Object:type nodes found
                      (Type 3 only)

Stage 2 — Answer generation quality (all conditions):
    Type 1 & 3: LLM-as-judge faithfulness score (1-5) + hallucination flag
    Type 2:     Binary precision / recall / F1 on anomaly label
                + deviation point accuracy

Output
------
results/
  eval_results_<condition>.jsonl   — per-question results per condition
  eval_summary.json                — aggregated metrics across all conditions

Usage
-----
    python run_evaluation.py \\
        --dataset eval_v2_combined.jsonl \\
        --graphml test2.graphml \\
        --faiss_db ./faiss_db_pm4py \\
        --model_id Qwen/Qwen2.5-7B-Instruct \\
        --judge_model gpt-4o \\
        --out_dir results \\
        --conditions rag,graphrag,unconstrained_gcr,constrained_gcr \\
        --constrained_paths predicted_paths_constrained.jsonl \\
        --unconstrained_paths predicted_paths_unconstrained.jsonl

    For quick testing with a smaller model:
        --model_id Qwen/Qwen2.5-1.5B-Instruct --limit 5

Predicted paths format (one line per question):
    {"id": "T1_001", "paths": ["Event:Create_Purchase_Order -> NEXT_FOR_purchase_order -> ..."]}

Dependencies
------------
    pip install transformers torch openai langchain-openai langchain-community
                faiss-cpu networkx tqdm python-dotenv
    Requires OPENAI_API_KEY in environment or .env file for the judge model.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import networkx as nx
import torch
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
judge_client = OpenAI()


# =============================================================================
# HF MODEL — answer generation
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
    system: str = None,
    max_new_tokens: int = 512,
) -> str:
    """
    Generate a response using the HF model's chat template.
    Works with Qwen2.5-Instruct and other instruction-tuned models.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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
# GRAPH UTILITIES
# =============================================================================

def load_graph(graphml_path: str) -> nx.DiGraph:
    G = nx.read_graphml(graphml_path)
    print(f"[graph] Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def get_local_subgraph_text(
    G: nx.DiGraph, topic_entities: List[str], max_depth: int = 2
) -> str:
    """
    GraphRAG condition: build text description of the local subgraph
    around the topic entities up to max_depth hops.
    """
    visited = set()
    lines = []

    def traverse(node, depth):
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
            rel = edata.get("label", "rel")
            lines.append(f"  --[{rel}]--> {nxt}")
            traverse(nxt, depth + 1)

    for ent in topic_entities:
        if ent in G:
            traverse(ent, 0)

    return "\n".join(lines) if lines else "No context found in graph."


def get_rag_context(
    question: str, faiss_db_path: str, k: int = 5
) -> str:
    """
    RAG condition: retrieve top-k documents from the FAISS index.
    Lazy-loads the retriever to avoid importing langchain at module level.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    vectorstore = FAISS.load_local(
        faiss_db_path,
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    docs = retriever.invoke(question)
    return "\n\n".join([doc.page_content for doc in docs])


def get_direct_answer(question_record: Dict, G: nx.DiGraph) -> str:
    """
    Direct graph query baseline: structured answer from graph data, no LLM.
    Upper bound on factual accuracy for structural questions.
    """
    q_type = question_record["type"]
    gt = question_record["ground_truth"]
    facts = gt.get("facts", {})

    if q_type == "lifecycle_explanation":
        seq = facts.get("event_sequence", [])
        lines = [
            f"Object {facts.get('object_type', '')} "
            f"{facts.get('object_id', '')} lifecycle:"
        ]
        for e in seq:
            lines.append(
                f"  [{e['timestamp']}] {e['activity']} (event {e['event_id']})"
            )
        return "\n".join(lines)

    elif q_type == "anomaly_detection":
        observed = facts.get("observed_variant", [])
        majority = facts.get("majority_variant", [])
        if observed == majority:
            return (
                f"The sequence follows the standard flow: {' -> '.join(majority)}"
            )
        else:
            deviation = gt.get("deviation", "Unknown deviation")
            return (
                f"The sequence deviates from the standard flow.\n"
                f"Standard: {' -> '.join(majority)}\n"
                f"Observed: {' -> '.join(observed)}\n"
                f"Deviation: {deviation}"
            )

    elif q_type == "cross_object_reasoning":
        obj_a = facts.get("object_a", {})
        obj_b = facts.get("object_b", {})
        lines = [
            f"Event {facts.get('event_id', '')} ({facts.get('activity', '')}) "
            f"at {facts.get('timestamp', '')}",
            f"\nObject A: {obj_a.get('object_type', '')} "
            f"{obj_a.get('object_id', '')}",
        ]
        for e in obj_a.get("event_sequence", []):
            lines.append(f"  [{e['timestamp']}] {e['activity']}")
        lines.append(
            f"\nObject B: {obj_b.get('object_type', '')} "
            f"{obj_b.get('object_id', '')}"
        )
        for e in obj_b.get("event_sequence", []):
            lines.append(f"  [{e['timestamp']}] {e['activity']}")
        return "\n".join(lines)

    return "No direct answer available."


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
# STAGE 1 — PATH HIT METRIC
# =============================================================================

def hit_metric_paths(predicted_paths: List[str], ground_truth: Dict) -> Dict:
    """
    Evaluate whether retrieved paths contain the ground truth entities.
    Expected nodes are formatted as Event:Activity_Name (underscored),
    matching the linearize_path() output format from gcr.py.

    Used for both constrained_gcr and unconstrained_gcr conditions so the
    impact of the Trie constraint on retrieval quality can be isolated.
    """
    pgt = ground_truth.get("path_ground_truth", {})
    expected_nodes = pgt.get("expected_nodes", [])
    object_nodes = pgt.get("object_nodes", [])

    if not expected_nodes and not object_nodes:
        return {"activity_recall": None, "note": "no path_ground_truth in record"}

    all_paths_text = " ".join(predicted_paths).lower()

    if expected_nodes:
        hits = [n for n in expected_nodes if n.lower() in all_paths_text]
        activity_recall = round(len(hits) / len(expected_nodes), 3)
        misses = [n for n in expected_nodes if n not in hits]
    else:
        activity_recall = None
        hits = []
        misses = []

    result = {
        "activity_recall": activity_recall,
        "hits": hits,
        "misses": misses,
    }

    if object_nodes:
        obj_hits = [n for n in object_nodes if n.lower() in all_paths_text]
        result["entity_recall"] = round(len(obj_hits) / len(object_nodes), 3)
        result["object_hits"] = obj_hits

    return result


# =============================================================================
# STAGE 2 — ANSWER EVALUATION METRICS
# =============================================================================

def judge_faithfulness(
    question: str,
    answer: str,
    ground_truth_facts: Dict,
    judge_model: str,
) -> Dict:
    """
    LLM-as-judge faithfulness scoring (1-5) using GPT-4o as an independent
    evaluator. The judge model is not the model under test.
    Returns {"score": int, "reasoning": str, "hallucination_flag": bool}
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
    The majority variant is provided in the question, so the model's answer
    is expected to be an explicit conformance/non-conformance statement.
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

    if predicted_anomalous and not predicted_normal:
        predicted = True
    elif predicted_normal and not predicted_anomalous:
        predicted = False
    else:
        predicted = predicted_anomalous

    correct = predicted == is_anomalous

    deviation_correct = False
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
        "correct": correct,
        "deviation_point_correct": deviation_correct if is_anomalous else None,
    }


# =============================================================================
# EVALUATION RUNNER
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
    predicted_paths_map: Optional[Dict[str, List[str]]] = None,
) -> List[Dict]:
    """
    Run evaluation for one condition.

    condition options:
        rag               — FAISS retrieval context + HF model
        graphrag          — local subgraph context + HF model
        unconstrained_gcr — free path generation context + HF model
        constrained_gcr   — Trie-constrained path generation context + HF model

    predicted_paths_map: dict of {question_id -> [path strings]} from the
        small LLM. Required for unconstrained_gcr and constrained_gcr.
        For constrained_gcr these are Trie-constrained paths.
        For unconstrained_gcr these are free-generation paths.
    """
    results = []
    print(f"\n[{condition.upper()}] Evaluating {len(questions)} questions...")

    # Validate inputs
    if condition == "rag" and not faiss_db_path:
        raise ValueError("--faiss_db is required for the rag condition.")
    if condition in ("unconstrained_gcr", "constrained_gcr") and not predicted_paths_map:
        raise ValueError(
            f"--{'constrained' if condition == 'constrained_gcr' else 'unconstrained'}"
            f"_paths is required for the {condition} condition."
        )

    for q in tqdm(questions):
        question_text = q["question"]
        topic_entities = q["topic_entities"]
        q_type = q["type"]
        gt = q["ground_truth"]

        # --- Build context and generate answer ---
        if condition == "rag":
            context = get_rag_context(question_text, faiss_db_path)
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
                # Fall back to no-context if paths are missing for this question
                prompt = build_noctx_prompt(question_text)
            answer = call_hf_model(prompt, tokenizer, hf_model, SYSTEM_PROCESS_MINING)

        else:
            raise ValueError(f"Unknown condition: {condition}")

        # --- Build result record ---
        result = {
            "id": q["id"],
            "type": q_type,
            "condition": condition,
            "question": question_text,
            "answer": answer,
        }

        # --- Stage 1: Path hit metric (GCR conditions only) ---
        if condition in ("unconstrained_gcr", "constrained_gcr"):
            predicted_paths = predicted_paths_map.get(q["id"], [])
            gt_with_type = {**gt, "type": q_type}
            result["path_hit_metric"] = hit_metric_paths(predicted_paths, gt_with_type)

        # --- Stage 2: Answer quality metrics ---
        if q_type == "anomaly_detection":
            result["binary_eval"] = evaluate_anomaly_detection(answer, gt)
            result["faithfulness"] = judge_faithfulness(
                question_text, answer, gt.get("facts", {}), judge_model
            )
        else:
            result["faithfulness"] = judge_faithfulness(
                question_text, answer, gt.get("facts", {}), judge_model
            )

        results.append(result)

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  Written: {out_path}")
    return results


# =============================================================================
# SUMMARY AGGREGATION
# =============================================================================

def compute_summary(all_results: Dict[str, List[Dict]]) -> Dict:
    """Aggregate metrics across all conditions into a single summary dict."""
    summary = {}

    for condition, results in all_results.items():
        cond = {"total": len(results)}

        # Faithfulness (Types 1 and 3)
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
            round(sum(faith_scores) / len(faith_scores), 3)
            if faith_scores else None
        )
        cond["hallucination_rate"] = (
            round(sum(hallucination_flags) / len(hallucination_flags), 3)
            if hallucination_flags else None
        )

        # Type 2: precision / recall / F1 / deviation accuracy
        t2 = [r for r in results if r["type"] == "anomaly_detection"]
        if t2:
            tp = sum(
                1 for r in t2
                if r["binary_eval"]["predicted_anomalous"]
                and r["binary_eval"]["actual_anomalous"]
            )
            fp = sum(
                1 for r in t2
                if r["binary_eval"]["predicted_anomalous"]
                and not r["binary_eval"]["actual_anomalous"]
            )
            fn = sum(
                1 for r in t2
                if not r["binary_eval"]["predicted_anomalous"]
                and r["binary_eval"]["actual_anomalous"]
            )
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

        # Stage 1: Path hit metric (GCR conditions only)
        path_hits = [r["path_hit_metric"] for r in results if "path_hit_metric" in r]
        if path_hits:
            act_recalls = [
                h["activity_recall"]
                for h in path_hits
                if h.get("activity_recall") is not None
            ]
            ent_recalls = [
                h["entity_recall"]
                for h in path_hits
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True,
        help="Path to eval_v2_combined.jsonl"
    )
    parser.add_argument(
        "--graphml", required=True,
        help="Path to test2.graphml"
    )
    parser.add_argument(
        "--faiss_db", default="./faiss_db_pm4py",
        help="Path to FAISS index directory (required for rag condition)"
    )
    parser.add_argument(
        "--model_id", default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID for answer generation across all conditions"
    )
    parser.add_argument(
        "--judge_model", default="gpt-4o",
        help="OpenAI model used as independent LLM judge (not the model under test)"
    )
    parser.add_argument(
        "--constrained_paths", default=None,
        help=(
            "JSONL file of Trie-constrained paths from the small LLM. "
            "Each line: {\"id\": \"T1_001\", \"paths\": [\"Event:Create_PO -> ...\"]}. "
            "Required for constrained_gcr condition."
        )
    )
    parser.add_argument(
        "--unconstrained_paths", default=None,
        help=(
            "JSONL file of unconstrained paths from the small LLM. "
            "Same format as --constrained_paths. "
            "Required for unconstrained_gcr condition."
        )
    )
    parser.add_argument(
        "--conditions",
        default="rag,graphrag,unconstrained_gcr,constrained_gcr",
        help="Comma-separated list of conditions to run"
    )
    parser.add_argument("--out_dir", default="results")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of questions (useful for quick testing)"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load graph
    G = load_graph(args.graphml)

    # Load dataset
    with open(args.dataset) as f:
        questions = [json.loads(line) for line in f]
    if args.limit:
        questions = questions[:args.limit]
    print(f"Loaded {len(questions)} questions.")

    # Load predicted paths files
    def load_paths_file(path: Optional[str]) -> Optional[Dict[str, List[str]]]:
        if not path or not os.path.exists(path):
            return None
        result = {}
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                result[rec["id"]] = rec.get("paths", [])
        print(f"Loaded predicted paths from {path} ({len(result)} entries).")
        return result

    constrained_paths_map = load_paths_file(args.constrained_paths)
    unconstrained_paths_map = load_paths_file(args.unconstrained_paths)

    # Load HF model once — shared across all conditions
    tokenizer, hf_model = load_hf_model(args.model_id)

    conditions = [c.strip() for c in args.conditions.split(",")]
    all_results = {}

    for condition in conditions:
        # Select the right paths map for GCR conditions
        if condition == "constrained_gcr":
            paths_map = constrained_paths_map
        elif condition == "unconstrained_gcr":
            paths_map = unconstrained_paths_map
        else:
            paths_map = None

        out_path = os.path.join(args.out_dir, f"eval_results_{condition}.jsonl")
        results = run_condition(
            questions=questions,
            condition=condition,
            G=G,
            tokenizer=tokenizer,
            hf_model=hf_model,
            judge_model=args.judge_model,
            out_path=out_path,
            faiss_db_path=args.faiss_db,
            predicted_paths_map=paths_map,
        )
        all_results[condition] = results

    # Aggregate and write summary
    summary = compute_summary(all_results)
    summary_path = os.path.join(args.out_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n========== EVALUATION SUMMARY ==========")
    print(json.dumps(summary, indent=2))
    print(f"\nWritten: {summary_path}")


if __name__ == "__main__":
    main()