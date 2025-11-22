
# docintel

Local AI system for document intelligence.

DocIntel reads PDFs and text files, classifies them (Invoice, Resume, Utility Bill, Other), extracts structured data using regular expressions, and provides semantic search using SentenceTransformers. Outputs clean JSON results for easy integration.

## Features
- Ingest PDFs and plain text files
- Classify documents into common categories (Invoice, Resume, Utility Bill, Other)
- Extract structured fields using configurable regular expressions
- Compute embeddings and perform semantic search with SentenceTransformers
- Produce JSON output suitable for downstream ingestion or APIs

## Quick Start

1. Clone the repository
   ```
   git clone https://github.com/faaizazizpf/docintel.git
   cd docintel
   ```

2. Create and activate a virtual environment (Pycharm is recommended)
   - macOS / Linux:
     ```
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

3. Install Python dependencies
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Notes:
   - For GPU usage, install a torch build compatible with your CUDA version (see https://pytorch.org).
   - You may need system libraries for PDF parsers (e.g., libpoppler) depending on your OS.

4. Run the program
   - In the repository root:
     ```
     python main.py
     ```

## Example: quick verification script
This minimal check ensures the embedding libraries work (not a repo script):
```
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-MiniLM-L6-v2'); v = m.encode(['hello world']); print(len(v), v[0][:6])"
```

## Main libraries used
- PDF & document parsing: pdfplumber, pdfminer.six, pypdfium2, Pillow
- Embeddings & semantic search: sentence-transformers, transformers, tokenizers, torch, safetensors, huggingface-hub
- ML & utilities: scikit-learn, numpy, scipy, joblib
- Text extraction/parsing: regex (regular expressions)
- Others: requests, fsspec, tqdm, PyYAML, packaging

## Output
- Produces JSON files containing:
  - Document category (Invoice/Resume/Utility Bill/Other)
  - Extracted structured fields (e.g., dates, invoice number, totals)
  - Embeddings or references used for semantic search (depending on configuration)


