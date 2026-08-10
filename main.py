"""
Entry point for the RAG-Agent application.

Supports two modes:
  CLI mode   — python main.py --preprocess / --index / --ask "question"
  Server mode — python main.py --serve   (or APP_MODE=api in the environment)

The CLI mode is unchanged from the original design, so existing workflows
continue to work without modification.
"""

from __future__ import annotations

import argparse
import sys

from config import Settings, ensure_data_dirs
from logging_config import configure_logging


def main() -> None:
    settings = Settings()
    configure_logging(log_level=settings.log_level, log_format=settings.log_format)

    parser = argparse.ArgumentParser(description="PDF OCR + RAG pipeline: preprocess, index, and query PDF documents.")
    parser.add_argument(
        "--preprocess", action="store_true", help="OCR and preprocess PDFs in the raw documents directory."
    )
    parser.add_argument(
        "--index", action="store_true", help="Build the vector index from preprocessed document artifacts."
    )
    parser.add_argument("--ask", type=str, metavar="QUESTION", help="Ask a question against the indexed corpus.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.default_top_k,
        help="Number of retrieved chunks to use for QA (default: %(default)s).",
    )
    parser.add_argument(
        "--force-preprocess", action="store_true", help="Re-run preprocessing even if artifacts already exist."
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI HTTP server instead of running a CLI command.",
    )
    args = parser.parse_args()

    if args.serve or settings.app_mode == "api":
        _run_server(settings)
        return

    if not args.preprocess and not args.index and not args.ask:
        parser.error("Specify at least one of: --preprocess, --index, --ask, --serve")

    ensure_data_dirs(settings)

    if args.preprocess:
        from document_process.pipeline import preprocess_document

        pdfs = sorted(p for p in settings.raw_documents_dir.iterdir() if p.suffix.lower() == ".pdf")
        if not pdfs:
            print(f"No PDF files found in {settings.raw_documents_dir}.", file=sys.stderr)
            sys.exit(1)
        for pdf_path in pdfs:
            result = preprocess_document(pdf_path, settings=settings, force=args.force_preprocess)
            print(f"preprocessed  {pdf_path.name}  →  {result.document_id}  ({result.chunk_count} chunks)")

    if args.index:
        from rag.retrieve import index_all_processed_documents

        store_path = settings.vectorstore_dir / "store.json"
        if store_path.exists():
            store_path.unlink()
        indexed = index_all_processed_documents(settings=settings)
        print(f"indexed {sum(indexed.values())} chunks across {len(indexed)} document(s)")

    if args.ask:
        from rag.dispatch import answer_question

        response = answer_question(args.ask, settings=settings, top_k=args.top_k)
        print("\nAnswer:")
        print(response.answer)
        if response.sources:
            print("\nSources:")
            for source in response.sources:
                print(
                    f"  chunk={source['chunk_id']}  page={source.get('page_number')}  "
                    f"score={source.get('score'):.4f}  file={source.get('source_filename')}  "
                    f"regions={source.get('region_ids')}"
                )


def _run_server(settings: Settings) -> None:
    """
    Start the FastAPI + uvicorn HTTP server.

    This is the entry point used in Docker and on AWS (ECS Fargate).
    uvicorn is a production-quality ASGI server.
    """
    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed. Run: pip install 'uvicorn[standard]'",
            file=sys.stderr,
        )
        sys.exit(1)

    from api.app import create_app

    app = create_app(settings=settings)
    uvicorn.run(
        app,
        host="0.0.0.0",  # noqa: S104  bind to all interfaces inside the container
        port=8000,
        log_config=None,  # disable uvicorn's own logging setup; we already configured it
    )


if __name__ == "__main__":
    main()
