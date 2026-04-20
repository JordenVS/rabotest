"""
rag.py — RAG pipeline for P2P OCEL process logs.

Embedding backends supported:
  - "openai"       : OpenAI text-embedding-3-small  (requires OPENAI_API_KEY)
  - "bge"          : BAAI/bge-large-en-v1.5          (HuggingFace, recommended baseline)
  - "minilm"       : all-MiniLM-L6-v2               (HuggingFace, lightweight baseline)
  - "e5"           : intfloat/e5-large-v2            (HuggingFace)

LLM backends supported:
  - "openai"       : any OpenAI-compatible chat model via ChatOpenAI
  - "hf"           : any HuggingFace causal LM via HuggingFacePipeline

Usage example:
    retriever = get_retriever(docs, "./faiss_db", embedding_backend="bge")
    chain     = create_rag_chain(retriever, llm_backend="hf", llm_model="Qwen/Qwen2.5-7B-Instruct")
    response  = chain.invoke("What is a normal process in the P2P system?")
    print(response["answer"])
"""

from __future__ import annotations

from typing import Literal

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
EmbeddingBackend = Literal["openai", "bge", "minilm", "e5"]
LLMBackend = Literal["openai", "hf"]

# ---------------------------------------------------------------------------
# Embedding model registry
# ---------------------------------------------------------------------------
_EMBEDDING_CONFIGS: dict[str, dict] = {
    "openai": {
        "model_name": "text-embedding-3-small",
        "description": "OpenAI text-embedding-3-small (proprietary)",
    },
    "bge": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "description": "BGE-large-en-v1.5 — strong open baseline (Muennighoff et al., 2023 MTEB)",
        "encode_kwargs": {"normalize_embeddings": True},
    },
    "minilm": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "all-MiniLM-L6-v2 — lightweight open baseline",
        "encode_kwargs": {"normalize_embeddings": True},
    },
    "e5": {
        "model_name": "intfloat/e5-large-v2",
        "description": "E5-large-v2 — Microsoft open baseline (Wang et al., 2022)",
        "encode_kwargs": {"normalize_embeddings": True},
    },
}


def _build_embeddings(backend: EmbeddingBackend):
    """Instantiate an embedding model by backend key."""
    cfg = _EMBEDDING_CONFIGS.get(backend)
    if cfg is None:
        raise ValueError(
            f"Unknown embedding backend '{backend}'. "
            f"Choose from: {list(_EMBEDDING_CONFIGS)}"
        )

    print(f"[Embeddings] Using: {cfg['description']}")

    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=cfg["model_name"])

    # HuggingFace / Sentence-Transformers path
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback

    return HuggingFaceEmbeddings(
        model_name=cfg["model_name"],
        encode_kwargs=cfg.get("encode_kwargs", {})
    )

from langchain_core.callbacks.base import BaseCallbackHandler

class TokenUsageCallback(BaseCallbackHandler):
    """Captures token usage from the last LLM call in a RAG chain."""
    def __init__(self):
        self.last_token_meta = {}

    def on_llm_end(self, response, **kwargs):
        # LangChain surfaces OpenAI usage in response.llm_output
        usage = response.llm_output.get("token_usage", {})
        self.last_token_meta = {
            "prompt_tokens_answer": usage.get("prompt_tokens"),
            "completion_tokens":    usage.get("completion_tokens"),
        }

# ---------------------------------------------------------------------------
# Vectorstore helpers
# ---------------------------------------------------------------------------

def get_retriever(
    docs,
    db_path: str,
    *,
    embedding_backend: EmbeddingBackend = "bge",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    k: int = 5,
    search_type: str = "similarity",
):
    """
    Chunk documents, embed them, persist a FAISS index, and return a retriever.

    Parameters
    ----------
    docs:
        LangChain or LlamaIndex documents (must have .page_content / .text).
    db_path:
        Local directory where the FAISS index will be saved.
    embedding_backend:
        One of "openai" | "bge" | "minilm" | "e5".
    chunk_size / chunk_overlap:
        RecursiveCharacterTextSplitter parameters.
    k:
        Number of chunks to retrieve per query.
    search_type:
        FAISS retriever search type ("similarity" or "mmr").
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunked_docs = splitter.split_documents(docs)
    print(f"[Retriever] {len(chunked_docs)} chunks from {len(docs)} documents.")

    embeddings = _build_embeddings(embedding_backend)

    print("[Retriever] Building FAISS index…")
    vectorstore = FAISS.from_documents(chunked_docs, embeddings)
    vectorstore.save_local(db_path)
    print(f"[Retriever] Index saved to '{db_path}'.")

    retriever = vectorstore.as_retriever(
        search_type=search_type, search_kwargs={"k": k}
    )
    print("[Retriever] Ready.")
    return retriever


def get_retriever_from_db(
    db_path: str,
    *,
    embedding_backend: EmbeddingBackend = "bge",
    k: int = 5,
    search_type: str = "similarity",
):
    """Load a previously saved FAISS index and return a retriever."""
    embeddings = _build_embeddings(embedding_backend)
    vectorstore = FAISS.load_local(
        db_path, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_type=search_type, search_kwargs={"k": k}
    )
    print(f"[Retriever] Loaded from '{db_path}'.")
    return retriever


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _build_llm(backend: LLMBackend, model: str, **kwargs):
    """Instantiate a chat/causal LLM by backend key."""
    if backend == "openai":
        from langchain_openai import ChatOpenAI
        print(f"[LLM] Using OpenAI model: {model}")
        return ChatOpenAI(model=model, **kwargs)

    if backend == "hf":
        from transformers import pipeline as hf_pipeline
        from langchain_huggingface import HuggingFacePipeline
        print(f"[LLM] Loading HuggingFace model: {model}")
        pipe = hf_pipeline(
            "text-generation",
            model=model,
   #         max_new_tokens=kwargs.get("max_new_tokens", 512),
            device_map="auto",
        )
        return HuggingFacePipeline(pipeline=pipe)

    raise ValueError(
        f"Unknown LLM backend '{backend}'. Choose from: 'openai' | 'hf'"
    )


# ---------------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------------

_RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a process mining assistant specialising in Procure-to-Pay (P2P) event logs.
Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so explicitly.
Keep your answer concise - one or two sentences.

Context:
{context}

Question:
{question}

Answer:"""
)


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(
    retriever,
    *,
    llm_backend: LLMBackend = "hf",
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
    **llm_kwargs,
):
    """
    Build a standard LangChain LCEL RAG chain.

    The chain accepts a dict {"question": <str>} and returns a dict with
    keys "question", "context", and "answer". The context is included in
    the output so that retrieval quality can be evaluated separately from
    generation quality (e.g. faithfulness, answer relevance).

    The retriever is called exactly once per invocation: the retrieved docs
    are stored under "context" and then reused by the prompt — avoiding the
    double-retrieval bug that arises when retriever | _format_docs appears
    in both the fetch step and the generation step.

    Parameters
    ----------
    retriever:
        A LangChain retriever (from get_retriever or get_retriever_from_db).
    llm_backend:
        "hf" for a local HuggingFace model, "openai" for the OpenAI API.
    llm_model:
        Model identifier passed to the chosen backend.
    **llm_kwargs:
        Extra kwargs forwarded to the LLM constructor (e.g. max_new_tokens).

    Example
    -------
    >>> chain = create_rag_chain(retriever, llm_backend="hf",
    ...                          llm_model="Qwen/Qwen2.5-7B-Instruct")
    >>> result = chain.invoke({"question": "What objects are linked to purchase_order:587?"})
    >>> print(result["answer"])
    """
    llm = _build_llm(llm_backend, llm_model, **llm_kwargs)

    # Step 1: fetch and format context once, pass question through unchanged.
    fetch = RunnablePassthrough.assign(
        context=lambda x: _format_docs(retriever.invoke(x["question"]))
    )

    # Step 2: format prompt from the already-fetched context + question.
    generate = (
        _RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    # Step 3: combine — keep question and context in the output dict for eval.
    full_chain = fetch | RunnablePassthrough.assign(answer=generate)

    print(
        f"[RAG Chain] Ready — LLM: {llm_backend}/{llm_model}"
    )
    return full_chain