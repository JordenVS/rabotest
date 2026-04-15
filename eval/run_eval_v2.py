"""
eval/run_evaluation.py
======================
Stage 2 of the two-stage GCR evaluation pipeline.

Loads predicted paths from generate_predicted_paths.py, generates final
natural-language answers using four systems, scores them, and writes a
results table.

Systems
-------
gcr_constrained_paths   — constrained GCR beam paths scored directly (no LLM)
gcr_unconstrained_paths — unconstrained GCR beam paths scored directly (no LLM)
gcr_constrained         — constrained GCR paths + enriched object context → large LLM
gcr_unconstrained       — unconstrained paths only → large LLM
rag                     — FAISS dense retrieval → large LLM
graphrag                — 1-hop subgraph context → large LLM

The *_paths systems evaluate the raw GCR output as a retrieval component,
independent of downstream LLM answer generation.  The concatenation of all
beam paths is used as the prediction string against the gold answer, and MRR
is computed over the individual beams.  This decouples path-retrieval quality
from generation quality, following the two-stage evaluation design of
Luo et al. (2025).

Metrics
-------
next_step      : Exact Match (EM), token F1, ROUGE-L F1, MRR (over GCR beams)
counterfactual : Binary accuracy (yes / no polarity)
all systems    : answer generation latency (mean, p95)
                 (latency is 0 for *_paths systems — no LLM is invoked)

Output files
------------
<out_dir>/answers.jsonl          one record per (instance, system)
<out_dir>/results_table.csv      aggregated per-system metrics
<out_dir>/results_table.tex      LaTeX table for the paper

Usage
-----
    python -m eval.run_evaluation \\
        --dataset              eval/sampled_100.json \\
        --constrained_paths    results/predicted_paths_constrained.jsonl \\
        --unconstrained_paths  results/predicted_paths_unconstrained.jsonl \\
        --graph_context        graphs/context_graph.graphml \\
        --faiss_db             faiss_db_bge \\
        --docs_cache           cache/pm4py_docs.pkl \\
        --llm_model            Qwen/Qwen2.5-7B-Instruct \\
        --emb_backend          bge \\
        --out_dir              results \\
        --device               cpu

    Evaluate only the path-retrieval quality (no LLM loaded):
        --path_metrics_only

    Skip path-only systems when running full evaluation:
        --skip_path_metrics

    Quick test (5 instances):
        --limit 5

    Skip slow systems:
        --skip_rag --skip_graphrag

References
----------
Luo, L., Zhao, Z., Haffari, G., Li, Y.-F., Gong, C., & Pan, S. (2025).
    Graph-constrained reasoning: Faithful reasoning on knowledge graphs
    with large language models. ICML 2025.
Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016).
    SQuAD: 100,000+ questions for machine comprehension of text. EMNLP.
Lin, C.-Y. (2004).
    ROUGE: A package for automatic evaluation of summaries.
    ACL Workshop on Text Summarisation Branches Out.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import string
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

import numpy as np
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ===========================================================================
# I/O helpers
# ===========================================================================

def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_dataset(path: str) -> List[Dict]:
    """Accepts JSON array or JSONL."""
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(l) for l in content.splitlines() if l.strip()]


def _load_done(path: str) -> set:
    """(instance_id, system) pairs already written — for resume support."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["instance_id"], r["system"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _append(path: str, record: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ===========================================================================
# Metrics  (Rajpurkar et al., 2016; Lin, 2004)
# ===========================================================================

def _normalise(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def _create_event(event_str: str) -> str:
    event_str = _normalise(event_str)
    event_str = event_str.replace(" ", "_")
    event_str =  "event:" + event_str if not event_str.startswith("event:") else event_str
    return event_str

def _tokens(text: str) -> List[str]:
    return _normalise(text).split()


def exact_match(prediction: str, gold: str) -> float:
    return float(_normalise(gold) in _normalise(prediction))


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 (Rajpurkar et al., 2016)."""
    pred_toks = _tokens(prediction)
    gold_toks = _tokens(gold)
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks)
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r)


def _lcs(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l(prediction: str, gold: str) -> float:
    """ROUGE-L F1 (Lin, 2004)."""
    p_toks = _tokens(prediction)
    g_toks = _tokens(gold)
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    lcs = _lcs(p_toks, g_toks)
    prec = lcs / len(p_toks)
    rec  = lcs / len(g_toks)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def mrr(beams: List[str], gold: str) -> float:
    """MRR over ranked beam outputs."""
    gold_norm = _normalise(gold)
    for rank, beam in enumerate(beams, start=1):
        if gold_norm in _normalise(beam):
            return 1.0 / rank
    return 0.0


def counterfactual_acc(prediction: str, gold_answer: str) -> float:
    """
    Binary accuracy for counterfactual questions.
    gold_answer is "Yes" or "No"; we check the first polar word in prediction.
    """
    p = _normalise(prediction)
    gold_polar = _normalise(gold_answer)
    has_no  = bool(re.search(r"\bno\b",  p))
    has_yes = bool(re.search(r"\byes\b", p))
    if gold_polar == "no":
        return float(has_no and not has_yes)
    return float(has_yes and not has_no)


def score_answer(
    prediction: str,
    gold_answer: str,
    question_family: str,
    beams: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute all applicable metrics for one (prediction, gold) pair."""
    scores: Dict[str, float] = {}
    if question_family == "next_step":
        scores["em"]      = exact_match(prediction, gold_answer)
        scores["tok_f1"]  = token_f1(prediction, gold_answer)
        scores["rouge_l"] = rouge_l(prediction, gold_answer)
        scores["mrr"]     = mrr(beams or [prediction], gold_answer)
    elif question_family == "counterfactual":
        scores["cf_acc"]  = counterfactual_acc(prediction, gold_answer)
        scores["em"]      = scores["cf_acc"]
    else:
        scores["em"] = exact_match(prediction, gold_answer)
    return scores


# ===========================================================================
# Path-only scoring  (no LLM invoked)
# ===========================================================================

def calculate_path_recall(predicted_beams: List[str], gold_paths: List[List[List[str]]]) -> float:
    """
    Measures how many of the gold path activity sequences are found within 
    the predicted beams.
    """
    if not gold_paths:
        return 1.0
    
    # Flatten gold paths into a list of required activity sequences
    # Gold format: [[["Act1", "Act2"]]] -> we check if "Act1" and "Act2" appear in order
    found_count = 0
    total_required = len(gold_paths)
    
    combined_predictions = " ".join(predicted_beams).lower() 
    for path_set in gold_paths:
        for sequence in path_set:
            # Check if all activities in the gold sequence appear in any predicted beam
            # (Simple heuristic: check if the activity names exist in the string)
            if all(_create_event(act) in combined_predictions for act in sequence):
                found_count += 1
                break        
    return found_count / total_required if total_required > 0 else 0.0

def score_paths_directly(
    beams: List[str],
    gold_answer: str,
    question_family: str,
    gold_paths: Optional[List] = None, # Added gold_paths for recall metrics
    context_block: str = ""           # Added to check context density
) -> Tuple[str, Dict[str, float]]:
    """
    Enhanced scoring for raw GCR beam paths and context blocks.
    """
    prediction = " ".join(b for b in beams if b).strip()
    
    # Base LLM metrics (EM, F1, ROUGE)
    scores = score_answer(prediction, gold_answer, question_family, beams=beams)
    
    # 1. Path Recall: Did the GCR retrieve the correct reasoning trace?
    if gold_paths:
        path_recall = calculate_path_recall(beams, gold_paths)
        print(f"Calculated path recall: {path_recall:.4f}")
        scores["path_recall"] = path_recall
    else:
        scores["path_recall"] = float("nan")

    # 2. Context Density: Ratio of useful content to total size
    # Measures if the context_block is concise or filled with 'noise'
    if context_block:
        total_chars = len(context_block)
        # Count occurrences of anchor object and related objects mentioned in gold
        # This is a proxy for 'signal'
        signal_count = context_block.count(":") # Rough proxy for attribute density
        scores["context_density"] = signal_count / (total_chars / 100) if total_chars > 0 else 0.0
        print(scores["context_density"])
    
    # 3. Path Precision: What % of beams contain gold activities?
    if gold_paths and beams:
        gold_acts = {act.lower() for path_set in gold_paths for seq in path_set for act in seq}
        hits = 0
        for b in beams:
            if any(ga in b.lower() for ga in gold_acts):
                hits += 1
        scores["path_precision"] = hits / len(beams)
        print(f"Calculated path precision: {scores['path_precision']:.4f}")

    return prediction, scores

# ===========================================================================
# LLM answer generation
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared prompt builder
# ---------------------------------------------------------------------------

_ANSWER_PROMPT = """\
You are a process mining assistant specialising in Procure-to-Pay (P2P) event logs.
Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so explicitly.
Keep your answer concise — one or two sentences.

### Context:
{context}

### Question:
{question}

### Answer:"""


def _build_answer_prompt(context: str, question: str) -> str:
    return _ANSWER_PROMPT.format(context=context.strip(), question=question.strip())


# ---------------------------------------------------------------------------
# GCR answer generation  (uses predicted paths + optional context_block)
# ---------------------------------------------------------------------------

def generate_gcr_answer(
    record: Dict,           # one predicted-path JSONL record
    question: str,
    llm,
) -> Tuple[str, float]:
    """
    Build the LLM prompt from the predicted paths and any enriched context,
    then call the LLM.

    The context block (when present) contains the structured object context
    produced by enrich_paths_with_context — provably grounded in the graph.
    Paths are listed first as the primary reasoning trace; the context block
    follows as supporting object detail.
    """
    paths: List[str] = record.get("paths", [])
    context_block: Optional[str] = record.get("context_block")

    path_text = "\n".join(
        f"  Path {i+1}: {p}" for i, p in enumerate(paths) if p
    ) or "  (no paths generated)"

    if context_block:
        context = (
            f"Reasoning paths (graph-constrained):\n{path_text}\n\n"
            f"Object context:\n{context_block}"
        )
    else:
        context = f"Reasoning paths:\n{path_text}"

    prompt = _build_answer_prompt(context, question)
    t0 = time.perf_counter()
    answer = llm(prompt)
    elapsed = time.perf_counter() - t0
    return answer, elapsed


# ---------------------------------------------------------------------------
# RAG answer generation
# ---------------------------------------------------------------------------

def generate_rag_answer(
    question: str,
    rag_chain,
) -> Tuple[str, float]:
    t0 = time.perf_counter()
    print(f"Generating RAG answer for question: {question}")
    result = rag_chain.invoke({"question": question})
    elapsed = time.perf_counter() - t0
    return result.get("answer", ""), elapsed


# ---------------------------------------------------------------------------
# GraphRAG answer generation
# ---------------------------------------------------------------------------

def generate_graphrag_answer(
    question: str,
    anchor_oid: str,
    graph,
    graphrag_llm,
) -> Tuple[str, float]:
    from graphrag.graphrag import perform_local_search
    t0 = time.perf_counter()
    answer = perform_local_search(graph, anchor_oid, question, llm=graphrag_llm)
    elapsed = time.perf_counter() - t0
    return answer, elapsed


# ---------------------------------------------------------------------------
# LLM factory  (mirrors rag.py pattern)
# ---------------------------------------------------------------------------

def build_llm_hf(model: str, device: str = "cpu", max_new_tokens: int = 256):
    """
    Return a callable: prompt_str -> answer_str.
    Uses a HuggingFace pipeline with chat-style inference when the tokenizer
    has a chat template, plain text-generation otherwise.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    print(f"[LLM] Loading answer-generation model: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype="auto",
        device_map=device,
    )
    pipe = pipeline(
        "text-generation",
        model=model_obj,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )
    print("[LLM] Ready.")

    def _call(prompt: str) -> str:
        # Use chat template if available (Qwen, Llama-3, etc.)
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            out = pipe(formatted)
            generated = out[0]["generated_text"]
            # Strip prompt prefix
            if generated.startswith(formatted):
                generated = generated[len(formatted):]
        else:
            out = pipe(prompt)
            generated = out[0]["generated_text"]
            if generated.startswith(prompt):
                generated = generated[len(prompt):]
        return generated.strip()

    return _call

def build_llm_openai(model: str, device: str = "cpu", max_new_tokens: int = 256):
    """
    Return a callable: prompt_str -> answer_str using OpenAI API.
    Note: 'device' is ignored as the model is hosted externally.
    """
    from openai import OpenAI
    
    # It will look for OPENAI_API_KEY in your environment variables
    client = OpenAI()

    print(f"[LLM] Initializing API-based model: {model}")

    def _call(prompt: str) -> str:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_new_tokens,
                temperature=0.0, # Recommended for consistent evaluation
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API Error: {e}")
            return ""

    return _call

def build_rag_chain(args):
    """Initialise the RAG retriever + chain."""
    from rag.rag import get_retriever, get_retriever_from_db, create_rag_chain

    if os.path.exists(args.faiss_db):
        print(f"[RAG] Loading FAISS index from {args.faiss_db}")
        from rag.rag import get_retriever_from_db
        retriever = get_retriever_from_db(
            args.faiss_db,
            embedding_backend=args.emb_backend,
            k=5,
        )
    else:
        print("[RAG] Building FAISS index from docs cache…")
        with open(args.docs_cache, "rb") as f:
            docs = pickle.load(f)
        from rag.rag import get_retriever
        retriever = get_retriever(
            docs,
            args.faiss_db,
            embedding_backend=args.emb_backend,
            k=5,
        )

    # RAG chain uses its own internal LLM instantiation
    from rag.rag import create_rag_chain
    chain = create_rag_chain(
        retriever,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
    )
    return chain


# ===========================================================================
# Aggregation
# ===========================================================================

def aggregate(scored: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Group per-instance score dicts by system, return mean metrics."""
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for s in scored:
        grouped[s["system"]].append(s)

    results = {}
    for system, recs in grouped.items():
        def _mean(key):
            vals = [r[key] for r in recs if key in r and r[key] is not None
                    and not (isinstance(r[key], float) and np.isnan(r[key]))]
            return float(np.mean(vals)) if vals else float("nan")

        def _p95(key):
            vals = [r[key] for r in recs if key in r and r[key] is not None
                    and not (isinstance(r[key], float) and np.isnan(r[key]))]
            return float(np.percentile(vals, 95)) if vals else float("nan")

        results[system] = {
            "n":           len(recs),
            "em":          _mean("em"),
            "tok_f1":      _mean("tok_f1"),
            "rouge_l":     _mean("rouge_l"),
            "mrr":         _mean("mrr"),
            "cf_acc":      _mean("cf_acc"),
            "lat_mean_s":  _mean("answer_s"),
            "lat_p95_s":   _p95("answer_s"),
        }
    return results


# ===========================================================================
# Main evaluation loop
# ===========================================================================

def run_evaluation(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    answers_path = os.path.join(args.out_dir, "answers.jsonl")
    done = set()
    #done = _load_done(answers_path)

    # ------------------------------------------------------------------ #
    # Load dataset
    # ------------------------------------------------------------------ #
    print(f"Loading dataset: {args.dataset}")
    questions = load_dataset(args.dataset)
    if args.limit:
        questions = questions[: args.limit]
        print(f"  [--limit] Using first {args.limit} instances.")

    # Index by instance_id for fast lookup
    q_index: Dict[str, Dict] = {q["instance_id"]: q for q in questions}
    print(f"  {len(questions)} instances loaded.\n")

    # ------------------------------------------------------------------ #
    # Load predicted paths (GCR)
    # ------------------------------------------------------------------ #
    constrained_index:   Dict[str, Dict] = {}
    unconstrained_index: Dict[str, Dict] = {}

    if args.constrained_paths and os.path.exists(args.constrained_paths):
        for rec in load_jsonl(args.constrained_paths):
            constrained_index[rec["instance_id"]] = rec
        print(f"Loaded {len(constrained_index)} constrained path records.")

    if args.unconstrained_paths and os.path.exists(args.unconstrained_paths):
        for rec in load_jsonl(args.unconstrained_paths):
            unconstrained_index[rec["instance_id"]] = rec
        print(f"Loaded {len(unconstrained_index)} unconstrained path records.\n")

    # ------------------------------------------------------------------ #
    # Load graph (GraphRAG needs it; GCR already used it in stage 1)
    # ------------------------------------------------------------------ #
    graph = None
    if not args.skip_graphrag:
        print(f"Loading context graph for GraphRAG: {args.graph_context}")
        from utils.graph_utils import load_graphml_to_networkx
        graph = load_graphml_to_networkx(args.graph_context)

    # ------------------------------------------------------------------ #
    # Build LLMs (lazy — only load what we need)
    # ------------------------------------------------------------------ #
    llm          = None
    rag_chain    = None
    graphrag_llm = None

    systems_needed = []
    # Path-only systems (no LLM required) — run first so they complete cheaply
    if constrained_index and not args.skip_path_metrics:
        systems_needed.append("gcr_constrained_paths")
    if unconstrained_index and not args.skip_path_metrics:
        systems_needed.append("gcr_unconstrained_paths")
    # LLM-backed systems — skipped entirely under --path_metrics_only
    if not args.path_metrics_only:
        if constrained_index:
            systems_needed.append("gcr_constrained")
        if unconstrained_index:
            systems_needed.append("gcr_unconstrained")
        if not args.skip_rag:
            systems_needed.append("rag")
        if not args.skip_graphrag:
            systems_needed.append("graphrag")

    print(f"Systems to evaluate: {systems_needed}\n")

    # ------------------------------------------------------------------ #
    # Evaluation loop — one system at a time to keep memory predictable
    # ------------------------------------------------------------------ #
    all_scored: List[Dict] = []

    for system in systems_needed:
        print(f"=== {system.upper()} ===")

        # --- lazy LLM init ---
        if system in ("gcr_constrained", "gcr_unconstrained") and llm is None:
            if args.llm_backend == "hf":
                llm = build_llm_hf(args.llm_model, args.device, args.max_new_tokens)
            elif args.llm_backend == "openai":
                llm = build_llm_openai(args.llm_model, args.device, args.max_new_tokens)

        if system == "rag" and rag_chain is None:
            rag_chain = build_rag_chain(args)

        if system == "graphrag" and graphrag_llm is None:
            from graphrag.graphrag import build_graphrag_llm
            if args.llm_backend == "hf":
                graphrag_llm = build_graphrag_llm(
                    backend="hf", model=args.llm_model,
                    max_new_tokens=args.max_new_tokens,
                )
            elif args.llm_backend == "openai":
                graphrag_llm = build_graphrag_llm(
                    backend="openai", model=args.llm_model,
                    max_new_tokens=args.max_new_tokens,
                )

        for q in tqdm(questions, desc=f"[{system}]"):
            instance_id = q["instance_id"]

            if (instance_id, system) in done:
                continue

            gold_answer    = q.get("gold_answer", "")
            gold_paths = q.get("gold_paths", []) # Extract gold_paths from sampled_100.json
            question_text  = q["question"]
            question_family = q.get("question_family", "unknown")
            anchor_oid     = q.get("anchor_object", {}).get("oid", "")

            # --- generate answer ---
            prediction = ""
            answer_s   = float("nan")
            beams: List[str] = []
            extra: Dict = {}

            try:
                # ... inside the loop: for q in tqdm(questions, desc=f"[{system}]"): ...
                if system == "gcr_constrained_paths":
                    path_rec = constrained_index.get(instance_id, {})
                    beams = path_rec.get("paths", [])
                    ctx = path_rec.get("context_block", "") # Extract context for density check
                    
                    # Call the modified scoring function
                    prediction, metrics = score_paths_directly(
                        beams, 
                        gold_answer, 
                        question_family, 
                        gold_paths=gold_paths, 
                        context_block=ctx
                    )
                    # ... rest of the record building ...
                    answer_s = 0.0   # no LLM latency
                    extra = {
                        "trie_build_s": path_rec.get("trie_build_s", 0.0),
                        "generation_s": path_rec.get("generation_s", 0.0),
                        "enrich_s":     path_rec.get("enrich_s", 0.0),
                        "path_total_s": path_rec.get("total_s", 0.0),
                    }
                    record = {
                        "instance_id":     instance_id,
                        "system":          system,
                        "question_family": question_family,
                        "question":        question_text,
                        "gold_answer":     gold_answer,
                        "prediction":      prediction,
                        "answer_s":        answer_s,
                        **metrics,
                        **extra,
                    }
                    _append(answers_path, record)
                    done.add((instance_id, system))
                    all_scored.append(record)
                    continue

                elif system == "gcr_unconstrained_paths":
                    # --- Path-only: score raw unconstrained beams, no LLM ---
                    path_rec = unconstrained_index.get(instance_id, {})
                    beams = path_rec.get("paths", [])
                    prediction, metrics = score_paths_directly(
                        beams, gold_answer, question_family
                    )
                    answer_s = 0.0
                    extra = {
                        "generation_s": path_rec.get("generation_s", 0.0),
                        "path_total_s": path_rec.get("total_s", 0.0),
                    }
                    record = {
                        "instance_id":     instance_id,
                        "system":          system,
                        "question_family": question_family,
                        "question":        question_text,
                        "gold_answer":     gold_answer,
                        "prediction":      prediction,
                        "answer_s":        answer_s,
                        **metrics,
                        **extra,
                    }
                    _append(answers_path, record)
                    done.add((instance_id, system))
                    all_scored.append(record)
                    continue

                elif system == "gcr_constrained":
                    path_rec = constrained_index.get(instance_id, {})
                    beams = path_rec.get("paths", [])
                    prediction, answer_s = generate_gcr_answer(
                        path_rec, question_text, llm
                    )
                    extra = {
                        "trie_build_s": path_rec.get("trie_build_s", 0.0),
                        "generation_s": path_rec.get("generation_s", 0.0),
                        "enrich_s":     path_rec.get("enrich_s", 0.0),
                        "path_total_s": path_rec.get("total_s", 0.0),
                    }

                elif system == "gcr_unconstrained":
                    path_rec = unconstrained_index.get(instance_id, {})
                    beams = path_rec.get("paths", [])
                    prediction, answer_s = generate_gcr_answer(
                        path_rec, question_text, llm
                    )
                    extra = {
                        "generation_s": path_rec.get("generation_s", 0.0),
                        "path_total_s": path_rec.get("total_s", 0.0),
                    }
                elif system == "rag":
                    prediction, answer_s = generate_rag_answer(
                        question_text, rag_chain
                    )
                    beams = [prediction]

                elif system == "graphrag":
                    if anchor_oid and graph is not None:
                        prediction, answer_s = generate_graphrag_answer(
                            question_text, anchor_oid, graph, graphrag_llm
                        )
                    else:
                        prediction = ""
                        answer_s   = 0.0
                    beams = [prediction]

            except Exception as exc:
                print(f"\n  [WARNING] {instance_id} / {system}: {exc}", flush=True)
                prediction = ""
                answer_s   = float("nan")

            # --- score ---
            metrics = score_answer(
                prediction, gold_answer, question_family, beams
            )

            record = {
                "instance_id":    instance_id,
                "system":         system,
                "question_family": question_family,
                "question":       question_text,
                "gold_answer":    gold_answer,
                "prediction":     prediction,
                "answer_s":       answer_s,
                **metrics,
                **extra,
            }

            _append(answers_path, record)
            done.add((instance_id, system))
            all_scored.append(record)

    # ------------------------------------------------------------------ #
    # Aggregate + report
    # ------------------------------------------------------------------ #
    # Load any previously written records not in this run's all_scored
    all_written = load_jsonl(answers_path) if os.path.exists(answers_path) else []
    results = aggregate(all_written)

    METRICS = ["n", "em", "tok_f1", "rouge_l", "mrr", "path_recall", "path_precision", "cf_acc",
               "lat_mean_s", "lat_p95_s"]

    print("\n========= RESULTS =========")
    header = f"{'System':<22}" + "".join(f"{m:>12}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for sys_name, vals in sorted(results.items()):
        row = f"{sys_name:<22}"
        for m in METRICS:
            v = vals.get(m, float("nan"))
            if isinstance(v, int):
                row += f"{v:>12d}"
            else:
                row += f"{v:>12.4f}"
        print(row)

    # CSV + LaTeX
    try:
        import pandas as pd
        df = (
            pd.DataFrame(results).T
            .reset_index()
            .rename(columns={"index": "system"})
        )
        csv_path = os.path.join(args.out_dir, "results_table.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nCSV  → {csv_path}")

        tex_path = os.path.join(args.out_dir, "results_table.tex")
        float_cols = [c for c in df.columns if c not in ("system", "n")]
        df[float_cols] = df[float_cols].map(
            lambda x: round(float(x), 4) if pd.notnull(x) else x
        )
        df.to_latex(
            tex_path,
            index=False,
            float_format="%.4f",
            caption=(
                "Evaluation results on 100-instance P2P OCEL benchmark. "
                "EM = exact match, MRR = mean reciprocal rank (GCR beams), "
                "CF-Acc = counterfactual accuracy, Lat = answer generation latency."
            ),
            label="tab:results",
        )
        print(f"LaTeX → {tex_path}")
    except ImportError:
        print("pandas not available — skipping CSV/LaTeX output.")

    print(f"\nAll answers → {answers_path}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage 2: generate final answers from GCR paths / RAG / GraphRAG "
            "and score them against gold annotations."
        )
    )
    # ---- inputs ----
    p.add_argument("--dataset", required=True,
                   help="Evaluation sample JSON/JSONL.")
    p.add_argument("--constrained_paths", default=None,
                   help="JSONL from generate_predicted_paths.py (constrained).")
    p.add_argument("--unconstrained_paths", default=None,
                   help="JSONL from generate_predicted_paths.py (unconstrained).")
    p.add_argument("--graph_context", default=None,
                   help="Context graph GraphML (required for GraphRAG).")
    p.add_argument("--faiss_db", default="./faiss_db_bge",
                   help="FAISS index directory (for RAG).")
    p.add_argument("--docs_cache", default="cache/pm4py_docs.pkl",
                   help="Pickled LangChain docs (used to build FAISS if index absent).")
    # ---- model ----
    p.add_argument("--llm_model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="HuggingFace model for answer generation.")
    p.add_argument("--llm_backend", default="openai",
                   choices=["hf", "openai"],
                   help="LLM backend for answer generation.")
    p.add_argument("--emb_backend", default="bge",
                   choices=["openai", "bge", "minilm", "e5"],
                   help="Embedding backend for RAG retriever.")
    p.add_argument("--device", default="cpu",
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--max_new_tokens", type=int, default=256)
    # ---- output ----
    p.add_argument("--out_dir", default="results",
                   help="Directory for answers.jsonl and results_table.*")
    # ---- control ----
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N instances (testing).")
    p.add_argument("--skip_rag", action="store_true",
                   help="Skip RAG system.")
    p.add_argument("--skip_graphrag", action="store_true",
                   help="Skip GraphRAG system.")
    p.add_argument("--skip_path_metrics", action="store_true",
                   help=(
                       "Skip the path-only systems (gcr_constrained_paths, "
                       "gcr_unconstrained_paths).  Useful when you only want "
                       "full end-to-end LLM evaluation."
                   ))
    p.add_argument("--path_metrics_only", action="store_true",
                   help=(
                       "Evaluate only the path-retrieval quality of the GCR "
                       "beam outputs against gold answers, without loading any "
                       "LLM.  Implies --skip_rag and --skip_graphrag and skips "
                       "gcr_constrained / gcr_unconstrained answer generation."
                   ))
    return p.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())