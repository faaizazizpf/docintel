
# docintel

Local AI system for document intelligence.

DocIntel reads PDFs and text files, classifies them (Invoice, Resume, Utility Bill, Other), extracts structured data using regular expressions, and provides semantic search using SentenceTransformers. Outputs clean JSON results for easy integration.

## Features
- Ingest PDFs and plain text files
- Classify documents into common categories (Invoice, Resume, Utility Bill, Other)
- Extract structured fields using configurable regular expressions
- Compute embeddings and perform semantic search with SentenceTransformers
- Produce JSON output suitable for downstream ingestion or APIs
- Complete local QA pipeline using RAG.

-------------------------------------

## Run Gemma 3 with Ollama (Windows Guide)

✅ 1. Install Ollama

   Download and install Ollama for Windows:
   
   ➡️ https://ollama.com/download
   
   After installation, Ollama runs a local server at:
   
   ```
   http://localhost:11434
   ```

✅ 2. Pull the Gemma 3 Model

Open PowerShell or CMD:

```
     ollama pull gemma3
```

Then while the ollama server runs, switch to/open new terminal

 ```
     ollama server
 ```
 

✅ 3. Test the Model (no Document knowledge)
Start a chat session with:
   ```
   ollama run gemma3
   ```
   To send a single prompt:
   ```
   ollama run gemma3 "Explain quantum computing in simple words."
   ```

✅ 4. Verify GPU Acceleration (Optional)

Ollama automatically uses your NVIDIA GPU if available.

Check GPU activity:
```
nvidia-smi
```
Look for ollama.exe using GPU memory.

To force-enable GPU on Windows PowerShell:

```
$env:OLLAMA_GPU="1"
```
Then run:

```
ollama run gemma3
```

✅ 5. Use Gemma 3 in Python

Install the Ollama Python client:
```
pip install ollama
```
Example script:
```
from ollama import Client

client = Client()
response = client.chat(
    model='gemma3',
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response['message']['content'])
```

===============================

## Quick Start

1. Clone the repository
   ```
   git clone https://github.com/faaizazizpf/docintel.git
   cd docintel
   ```

2. Create and activate a virtual environment (Pycharm is recommended)
   If you're not using Pycharm, you can do manually as follows:
   - Windows (PowerShell):
     ```
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
     
   - macOS / Linux:
     ```
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. Install Python dependencies
   Open project Terminal
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
- Bot Answer for user query


