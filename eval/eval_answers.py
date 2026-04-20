from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eval.metrics import (
    load_jsonl,
    load_dataset,
    load_done,
    append_record,
    score_answer,
    aggregate,
    print_results_table,
    write_results_table,
    ANSWER_METRIC_COLS,
)


# ===========================================================================
# LLM answer generation
# ===========================================================================

# _ANSWER_PROMPT = """\
# You are a process mining assistant specialising in Procure-to-Pay (P2P) event logs.
# Answer the question using ONLY the context provided below.
# If the context does not contain enough information, say so explicitly.
# Keep your answer concise — one or two sentences.

# ### Context:
# {context}

# ### Question:
# {question}

# ### Answer:"""

_ANSWER_PROMPT = """\
You are a process mining assistant specialising in Procure-to-Pay (P2P) event logs.
Answer the question using the context provided below.
Assume the context is accurate and complete, if something is not mentioned in there, it did not happen.
Keep your answer concise — one or two sentences.

### Context:
{context}

### Question:
{question}

### Answer:"""


def _build_answer_prompt(context: str, question: str) -> str:
    return _ANSWER_PROMPT.format(context=context.strip(), question=question.strip())


def generate_gcr_answer(
    record: Dict,
    question: str,
    llm,
) -> Tuple[str, float, dict, str]:
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
    answer, token_meta = llm(prompt)
    elapsed = time.perf_counter() - t0
    return answer, elapsed, token_meta, context


def generate_rag_answer(question, rag_chain) -> Tuple[str, float, dict, str]:
    from rag.rag import TokenUsageCallback
    callback = TokenUsageCallback()
    t0 = time.perf_counter()
    result = rag_chain.invoke(
        {"question": question},
        config={"callbacks": [callback]},
    )
    context = result.get("context", "")
    elapsed = time.perf_counter() - t0
    return result.get("answer", ""), elapsed, callback.last_token_meta, context


def generate_graphrag_answer(
    question: str,
    anchor_oid: str,
    graph,
    graphrag_llm,
) -> Tuple[str, float, dict, str]:
    from graphrag.graphrag import perform_local_search
    t0 = time.perf_counter()
    answer, token_meta, context = perform_local_search(graph, anchor_oid, question, llm=graphrag_llm, max_hops=2)
    elapsed = time.perf_counter() - t0
    return answer, elapsed, token_meta, context


# ===========================================================================
# LLM factories
# ===========================================================================

def build_llm_hf(model: str, device: str = "cpu", max_new_tokens: int = 256):
    """
    Return a callable: prompt_str -> answer_str.
    Uses the chat template when available (Qwen, Llama-3, etc.).
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
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            out = pipe(formatted)
            generated = out[0]["generated_text"]
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
    Return a callable: prompt_str -> answer_str using the OpenAI API.
    *device* is ignored — the model is hosted externally.
    """
    from openai import OpenAI

    client = OpenAI()
    print(f"[LLM] Initializing OpenAI model: {model}")

    def _call(prompt: str) -> str:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=max_new_tokens,
                temperature=0.0,
            )
            token_meta = {
                    "prompt_tokens_answer":  response.usage.prompt_tokens,
                    "completion_tokens":     response.usage.completion_tokens,
                }
            return response.choices[0].message.content.strip(), token_meta
        except Exception as e:
            print(f"API Error: {e}")
            return ""

    return _call


def build_rag_chain(args: argparse.Namespace):
    """Initialise the RAG retriever + chain."""
    from rag.rag import create_rag_chain

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

    chain = create_rag_chain(
        retriever,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
    )
    return chain


# ===========================================================================
# Main evaluation loop
# ===========================================================================

def run_answer_evaluation(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    answers_path = os.path.join(args.out_dir, "answers.jsonl")
    done = load_done(answers_path)

    # ------------------------------------------------------------------ #
    # Load dataset
    # ------------------------------------------------------------------ #
    print(f"Loading dataset: {args.dataset}")
    questions = load_dataset(args.dataset)
    if args.limit:
        questions = questions[: args.limit]
        print(f"  [--limit] Using first {args.limit} instances.")
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
    # Load graph (GraphRAG only)
    # ------------------------------------------------------------------ #
    graph = None
    if not args.skip_graphrag and args.graph_context:
        print(f"Loading context graph for GraphRAG: {args.graph_context}")
        from utils.graph_utils import load_graphml_to_networkx
        graph = load_graphml_to_networkx(args.graph_context)

    # ------------------------------------------------------------------ #
    # Systems to run
    # ------------------------------------------------------------------ #
    systems_needed: List[str] = []
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
    llm:          object = None
    rag_chain:    object = None
    graphrag_llm: object = None
    all_scored:   List[Dict] = []

    for system in systems_needed:
        print(f"=== {system.upper()} ===")

        # --- lazy LLM init ---
        if system in ("gcr_constrained", "gcr_unconstrained") and llm is None:
            if args.llm_backend == "hf":
                llm = build_llm_hf(args.llm_model, args.device, args.max_new_tokens)
            else:
                llm = build_llm_openai(args.llm_model, args.device, args.max_new_tokens)

        if system == "rag" and rag_chain is None:
            rag_chain = build_rag_chain(args)

        if system == "graphrag" and graphrag_llm is None:
            from graphrag.graphrag import build_graphrag_llm
            graphrag_llm = build_graphrag_llm(
                backend=args.llm_backend,
                model=args.llm_model,
                max_new_tokens=args.max_new_tokens,
            )

        for q in tqdm(questions, desc=f"[{system}]"):
            instance_id     = q["instance_id"]
            if (instance_id, system) in done:
                continue

            gold_answer     = q.get("gold_answer", "")
            question_text   = q["question"]
            question_family = q.get("question_family", "unknown")
            anchor_oid      = q.get("anchor_object", {}).get("oid", "")

            prediction = ""
            answer_s   = float("nan")
            beams: List[str] = []
            extra: Dict = {}

            print(f"\n[Instance {instance_id}] {question_text}")

            try:
                if system == "gcr_constrained":
                    path_rec   = constrained_index.get(instance_id, {})
                    beams      = path_rec.get("paths", [])
                    prediction, answer_s, token_meta, context = generate_gcr_answer(  
                        path_rec, question_text, llm
                    )
                    extra = {
                        "trie_build_s": path_rec.get("trie_build_s", 0.0),
                        "generation_s": path_rec.get("generation_s", 0.0),
                        "enrich_s":     path_rec.get("enrich_s", 0.0),
                        "path_total_s": path_rec.get("total_s", 0.0),
                        **token_meta,
                    }

                elif system == "gcr_unconstrained":
                    path_rec   = unconstrained_index.get(instance_id, {})
                    beams      = path_rec.get("paths", [])
                    prediction, answer_s, token_meta, context = generate_gcr_answer(  
                        path_rec, question_text, llm
                    )
                    extra = {
                        "generation_s": path_rec.get("generation_s", 0.0),
                        "path_total_s": path_rec.get("total_s", 0.0),
                        **token_meta,
                    }

                elif system == "rag":
                    prediction, answer_s, token_meta, context = generate_rag_answer(  
                        question_text, rag_chain
                    )
                    beams = [prediction]
                    extra = {**token_meta}

                elif system == "graphrag":
                    if anchor_oid and graph is not None:
                        prediction, answer_s, token_meta, context = generate_graphrag_answer(  
                            question_text, anchor_oid, graph, graphrag_llm
                        )
                    else:
                        prediction = ""
                        answer_s   = 0.0
                        token_meta = {"prompt_tokens_answer": None, "completion_tokens": None}
                    beams = [prediction]
                    extra = {**token_meta}
            except Exception as exc:
                print(f"\n  [WARNING] {instance_id} / {system}: {exc}", flush=True)
                prediction = ""
                answer_s   = float("nan")

            metrics = score_answer(
                prediction, gold_answer, question_family, beams
            )

            record = {
                "instance_id":     instance_id,
                "system":          system,
                "question_family": question_family,
                "question":        question_text,
                "gold_answer":     gold_answer,
                "prediction":      prediction,
                "context":         context,
                "answer_s":        answer_s,
                **metrics,
                **extra,
            }

            append_record(answers_path, record)
            done.add((instance_id, system))
            all_scored.append(record)

    # ------------------------------------------------------------------ #
    # Aggregate + report
    # ------------------------------------------------------------------ #
    all_written = load_jsonl(answers_path) if os.path.exists(answers_path) else []
    results = aggregate(all_written)

    print_results_table(results, ANSWER_METRIC_COLS)

    write_results_table(
        results,
        args.out_dir,
        caption=(
            "Answer-generation evaluation on 100-instance P2P OCEL benchmark. "
            "EM = exact match, MRR = mean reciprocal rank (GCR beams), "
            "CF-Acc = counterfactual accuracy, Lat = answer generation latency."
        ),
        label="tab:answer_results",
    )

    print(f"\nAll answers → {answers_path}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage 2b: generate final answers from GCR paths / RAG / GraphRAG "
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
                   help="Pickled LangChain docs (used to rebuild FAISS if absent).")
    # ---- model ----
    p.add_argument("--llm_model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="Model identifier for answer generation.")
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
                   help="Evaluate only the first N instances (for testing).")
    p.add_argument("--skip_rag", action="store_true",
                   help="Skip the RAG system.")
    p.add_argument("--skip_graphrag", action="store_true",
                   help="Skip the GraphRAG system.")
    return p.parse_args()


if __name__ == "__main__":
    run_answer_evaluation(parse_args())