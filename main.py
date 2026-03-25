from __future__ import annotations

import argparse
from pathlib import Path

from config import Settings
from document_Process.pipeline import preprocess_document
from rag.retrieve import answer_corpus_question, index_all_processed_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the PDF RAG index and ask a grounded question.")
    parser.add_argument("--build", action="store_true", help="Preprocess and index all PDFs in Data/Raw.")
    parser.add_argument("--ask", type=str, help="Ask a question against the indexed corpus.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of retrieved chunks to use for QA.")
    args = parser.parse_args()

    if not args.build and not args.ask:
        parser.error("Use --build, --ask, or both.")

    settings = Settings()

    if args.build:
        pdfs = sorted(path for path in settings.raw_documents_dir.iterdir() if path.suffix.lower() == ".pdf")
        if not pdfs:
            raise RuntimeError(f"No PDF files found in {settings.raw_documents_dir}.")
        for pdf_path in pdfs:
            result = preprocess_document(pdf_path, settings=settings)
            print(f"processed {pdf_path.name} -> {result.document_id} ({result.chunk_count} chunks)")
        indexed = index_all_processed_documents(settings=settings)
        print(f"indexed {sum(indexed.values())} chunks across {len(indexed)} document(s)")

    if args.ask:
        response = answer_corpus_question(args.ask, settings=settings, top_k=args.top_k)
        print("\nAnswer:")
        print(response.answer)
        if response.sources:
            print("\nSources:")
            for source in response.sources:
                print(
                    f"- chunk={source['chunk_id']} page={source.get('page_number')} "
                    f"score={source.get('score')} file={source.get('source_filename')}"
                )


if __name__ == "__main__":
    main()
