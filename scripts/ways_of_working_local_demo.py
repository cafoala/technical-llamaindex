#!/usr/bin/env python3
"""Local interactive RAG demo for ways-of-working docs.

What this script does:
- Loads markdown docs from data/ways-of-working
- Builds and persists a vector index (embeddings saved locally)
- Reuses the persisted index on later runs
- Uses reranking + routing for final answer generation
- Provides an interactive local Q&A loop

Based on prior git demos:
- 03_local_ways_of_working_rag.ipynb
- 06_demo_team_ops_from_saved_index_colab.ipynb
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive local RAG demo with persisted embeddings for ways-of-working docs."
    )
    parser.add_argument(
        "--docs-dir",
        default="data/ways-of-working",
        help="Directory containing source markdown documents.",
    )
    parser.add_argument(
        "--persist-dir",
        default="data/indexes/ways_of_working_nomic",
        help="Directory where the vector index is persisted.",
    )
    parser.add_argument(
        "--llm-model",
        default="llama3.2:3b",
        help="Ollama model used for generation.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=128,
        help="Maximum generated tokens per answer (lower is usually faster).",
    )
    parser.add_argument(
        "--embed-model",
        default="nomic-embed-text",
        help="Ollama embedding model.",
    )
    parser.add_argument(
        "--similarity-top-k",
        type=int,
        default=6,
        help="Number of initial vector candidates before reranking.",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=2,
        help="Number of chunks kept after reranking.",
    )
    parser.add_argument(
        "--reranking",
        choices=["on", "off"],
        default="off",
        help="Enable or disable LLM reranking. Disable for lower latency.",
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=8,
        help="Reranker batch size. Higher is often faster on small corpora.",
    )
    parser.add_argument(
        "--routing",
        choices=["on", "off"],
        default="off",
        help="Enable or disable routing. Disable for fastest factual Q&A.",
    )
    parser.add_argument(
        "--router-selector",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Router selector mode. Heuristic avoids an extra LLM routing call.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuilding persisted embeddings from source docs.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Apply faster settings: routing off, reranking off, "
            "similarity_top_k<=4, rerank_top_n<=2, rerank_batch_size>=8."
        ),
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Ask one question and exit (non-interactive mode).",
    )
    return parser.parse_args()


def configure_models(llm_model: str, embed_model: str, num_predict: int) -> None:
    # Keep notebook-like runs readable by suppressing verbose HTTP logs.
    for noisy_logger in ("httpx", "httpcore"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Limit generated tokens to keep interactive answers snappy.
    Settings.llm = Ollama(
        model=llm_model,
        request_timeout=180.0,
        additional_kwargs={"num_predict": num_predict},
    )
    Settings.embed_model = OllamaEmbedding(model_name=embed_model)
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50


def apply_fast_preset(args: argparse.Namespace) -> None:
    if not args.fast:
        return

    args.routing = "off"
    args.reranking = "off"
    args.similarity_top_k = min(args.similarity_top_k, 2)
    args.rerank_top_n = min(args.rerank_top_n, 2)
    args.rerank_batch_size = max(args.rerank_batch_size, 8)
    args.num_predict = min(args.num_predict, 96)
    if args.routing == "on":
        args.router_selector = "heuristic"


def discover_source_files(docs_dir: Path) -> list[str]:
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    source_files = sorted(str(p) for p in docs_dir.glob("*.md"))
    if not source_files:
        raise FileNotFoundError(f"No markdown files found in: {docs_dir}")
    return source_files


def load_documents(source_files: list[str]):
    documents = SimpleDirectoryReader(input_files=source_files).load_data()
    if not documents:
        raise ValueError("No documents loaded from source files.")
    return documents


def load_or_build_index(
    source_files: list[str],
    persist_dir: Path,
    rebuild_index: bool,
    include_documents: bool,
):
    has_vector_store = bool(list(persist_dir.glob("*_vector_store.json")))
    has_index = (
        persist_dir.exists()
        and (persist_dir / "docstore.json").exists()
        and (persist_dir / "index_store.json").exists()
        and has_vector_store
    )

    if has_index and not rebuild_index:
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        index = load_index_from_storage(storage_context)
        status = f"Loaded existing index from {persist_dir}"
        documents = load_documents(source_files) if include_documents else None
        return index, documents, status

    documents = load_documents(source_files)
    index = VectorStoreIndex.from_documents(documents)

    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))
    status = f"Built and persisted index to {persist_dir}"
    return index, documents, status


def build_fact_engine(
    index,
    similarity_top_k: int,
    use_reranking: bool,
    rerank_top_n: int,
    rerank_batch_size: int,
):
    if not use_reranking:
        return index.as_query_engine(similarity_top_k=similarity_top_k)

    reranker = LLMRerank(choice_batch_size=rerank_batch_size, top_n=rerank_top_n)
    return index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=[reranker],
    )


def build_llm_router(fact_engine, documents):
    if not documents:
        raise ValueError("Documents are required to build the summary tool for LLM routing.")

    # Summary tool: broad synthesis over the full document set.
    summary_index = SummaryIndex.from_documents(documents)
    summary_engine = summary_index.as_query_engine()

    fact_tool = QueryEngineTool.from_defaults(
        query_engine=fact_engine,
        description=(
            "Use for factual, specific, policy, process, or checklist questions "
            "about ways-of-working documents."
        ),
    )
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_engine,
        description=(
            "Use for high-level summaries, overviews, and synthesis across all "
            "ways-of-working documents."
        ),
    )

    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[summary_tool, fact_tool],
    )
    return router


def should_use_summary_route(question: str) -> bool:
    q = question.lower()
    summary_markers = [
        "summar",
        "overview",
        "high-level",
        "big picture",
        "main themes",
        "across all",
        "overall",
    ]
    return any(marker in q for marker in summary_markers)


def query_with_heuristic_router(
    question: str,
    fact_engine,
    get_summary_engine,
):
    if should_use_summary_route(question):
        return "summary", get_summary_engine().query(question)
    return "q&a", fact_engine.query(question)


def print_response(question: str, response, route_label: str | None = None) -> None:
    print("\n=== Question ===")
    print(question)
    print("\n=== Answer ===")
    print(response)

    selector = None
    if hasattr(response, "metadata") and isinstance(response.metadata, dict):
        selector = response.metadata.get("selector_result")

    if selector is not None:
        print("\n=== Router Selection ===")
        print(selector)
    elif route_label is not None:
        print("\n=== Router Selection ===")
        print(f"heuristic route: {route_label}")

    if hasattr(response, "source_nodes") and response.source_nodes:
        print("\n=== Source Chunks Used ===")
        for i, node in enumerate(response.source_nodes, 1):
            score = getattr(node, "score", None)
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
            preview = node.text[:160].replace("\n", " ")
            print(f"[{i}] score={score_str} | {preview}...")


def main() -> int:
    args = parse_args()
    apply_fast_preset(args)

    docs_dir = Path(args.docs_dir)
    persist_dir = Path(args.persist_dir)

    configure_models(args.llm_model, args.embed_model, args.num_predict)

    source_files = discover_source_files(docs_dir)
    include_documents = args.routing == "on" and args.router_selector == "llm"

    index, documents, status = load_or_build_index(
        source_files=source_files,
        persist_dir=persist_dir,
        rebuild_index=args.rebuild_index,
        include_documents=include_documents,
    )

    fact_engine = build_fact_engine(
        index=index,
        similarity_top_k=args.similarity_top_k,
        use_reranking=args.reranking == "on",
        rerank_top_n=args.rerank_top_n,
        rerank_batch_size=args.rerank_batch_size,
    )

    router = None
    summary_engine = None

    def get_summary_engine():
        nonlocal summary_engine, documents
        if summary_engine is not None:
            return summary_engine
        if documents is None:
            documents = load_documents(source_files)
        summary_index = SummaryIndex.from_documents(documents)
        summary_engine = summary_index.as_query_engine()
        return summary_engine

    if args.routing == "on" and args.router_selector == "llm":
        router = build_llm_router(
            fact_engine=fact_engine,
            documents=documents,
        )

    print(status)
    print(f"Loaded {len(source_files)} source file(s) from {docs_dir}")
    routing_mode = args.routing if args.routing == "off" else f"{args.routing}/{args.router_selector}"
    print(
        "Ready. Ask a question, or type 'exit' / 'quit' to stop. "
        f"(routing={routing_mode}, reranking={args.reranking}, "
        f"similarity_top_k={args.similarity_top_k}, rerank_top_n={args.rerank_top_n}, "
        f"rerank_batch_size={args.rerank_batch_size}, num_predict={args.num_predict})"
    )

    if args.question:
        t0 = perf_counter()
        if args.routing == "off":
            route_label, response = None, fact_engine.query(args.question)
        elif args.router_selector == "llm":
            route_label, response = None, router.query(args.question)
        else:
            route_label, response = query_with_heuristic_router(
                args.question,
                fact_engine=fact_engine,
                get_summary_engine=get_summary_engine,
            )
        elapsed = perf_counter() - t0
        print_response(args.question, response, route_label=route_label)
        print(f"\n=== Query Time ===\n{elapsed:.2f}s")
        return 0

    while True:
        question = input("\nYou> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return 0

        t0 = perf_counter()
        if args.routing == "off":
            route_label, response = None, fact_engine.query(question)
        elif args.router_selector == "llm":
            route_label, response = None, router.query(question)
        else:
            route_label, response = query_with_heuristic_router(
                question,
                fact_engine=fact_engine,
                get_summary_engine=get_summary_engine,
            )
        elapsed = perf_counter() - t0
        print_response(question, response, route_label=route_label)
        print(f"\n=== Query Time ===\n{elapsed:.2f}s")


if __name__ == "__main__":
    raise SystemExit(main())
