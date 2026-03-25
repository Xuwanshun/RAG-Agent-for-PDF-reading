# PDF RAG

Process PDFs from `Data/Raw`, build processed artifacts in `Data/Processed`, store embeddings in `Data/Embedded`, and answer grounded questions from retrieved document context.

Required environment variable:
- `OPENAI_API_KEY` (may be set in the OS environment or a local `.env` file)

Put raw PDFs in:
- `Data/Raw`

Run:
```bash
python main.py --build
python main.py --ask "What does the document say about AI risk management?"
```

Example:
```bash
python main.py --build --ask "What is the goal of the AI RMF?"
```
