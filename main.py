from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import Settings
from document_Process.pipeline import preprocess_document
from rag.qa import answer_question_from_frozen_artifacts
from rag.retrieve import index_all_processed_documents


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Frozen PDF preprocessing, indexing, and grounded QA.")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess PDFs in Data/Raw into frozen artifacts.")
    parser.add_argument("--index", action="store_true", help="Build the vector index from frozen processed outputs.")
    parser.add_argument("--ask", type=str, help="Ask a question against the indexed corpus.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of retrieved chunks to use for QA.")
    parser.add_argument("--force-preprocess", action="store_true", help="Rerun preprocessing even if frozen artifacts already exist.")
    args = parser.parse_args()

    if not args.preprocess and not args.index and not args.ask:
        parser.error("Use --preprocess, --index, --ask, or a combination.")

    settings = Settings()

    if args.preprocess:
        pdfs = sorted(path for path in settings.raw_documents_dir.iterdir() if path.suffix.lower() == ".pdf")
        if not pdfs:
            raise RuntimeError(f"No PDF files found in {settings.raw_documents_dir}.")
        for pdf_path in pdfs:
            result = preprocess_document(pdf_path, settings=settings, force=args.force_preprocess)
            print(f"processed {pdf_path.name} -> {result.document_id} ({result.chunk_count} chunks)")

    if args.index:
        store_path = settings.vectorstore_dir / "store.json"
        if store_path.exists():
            store_path.unlink()
        indexed = index_all_processed_documents(settings=settings)
        print(f"indexed {sum(indexed.values())} chunks across {len(indexed)} document(s)")

    if args.ask:
        response = answer_question_from_frozen_artifacts(args.ask, settings=settings, top_k=args.top_k)
        print("\nAnswer:")
        print(response.answer)
        if response.sources:
            print("\nSources:")
            for source in response.sources:
                print(
                    f"- chunk={source['chunk_id']} page={source.get('page_number')} "
                    f"score={source.get('score')} file={source.get('source_filename')} "
                    f"regions={source.get('region_ids')}"
                )


if __name__ == "__main__":
    main()
