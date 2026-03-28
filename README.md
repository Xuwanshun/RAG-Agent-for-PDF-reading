# PDF RAG

Process PDFs from `Data/Raw`, freeze processed artifacts in `Data/Processed`, build embeddings in `Data/Embedded`, and answer grounded questions from the saved document package and index.

## Requirements

- Linux with Python 3.11
- `OPENAI_API_KEY` in the shell environment or a local `.env` file for `--index` and `--ask`

Optional:
- `OPENAI_BASE_URL` if you are using an OpenAI-compatible endpoint
- `chromadb` only if you want to switch from the built-in JSON store to Chroma

## Local Linux Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set your environment values in `.env`:

```env
OPENAI_API_KEY=
OPENAI_BASE_URL=
```

## Project Layout

- Put source PDFs in `Data/Raw`
- Processed artifacts are written to `Data/Processed`
- Embeddings are written to `Data/Embedded`

These directories are created automatically when the app runs.

## Run

Show CLI help:

```bash
python main.py --help
```

Freeze preprocessing outputs:

```bash
python main.py --preprocess
```

Build embeddings from frozen chunks:

```bash
python main.py --index
```

Ask a question against the frozen index and artifacts:

```bash
python main.py --ask "What is the goal of the AI RMF?"
```

## Detection Notes

The preprocessing pipeline uses:

- PDF page rendering from `pypdfium2`
- PaddleOCR for OCR text extraction, bounding boxes, and confidence scores
- Paddle `LayoutDetection` (`PP-DocLayout_plus-L`) for text blocks, tables, and figure/image regions
- Reading order based on OCR text boxes only
- Frozen visual summaries derived from saved OCR/layout/chunk outputs

Important limitations:

- The first run downloads official Paddle models into `.paddlex/`
- Figure detection comes from the layout detector's real visual labels such as `image` and `figure`; there is no separate chart classifier in the active path
- The only LLM backend used for indexing/query-time reasoning is OpenAI
