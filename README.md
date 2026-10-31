# DocuMind AI

An Intelligent Document Analysis and Question-Answering System
Using Retrieval-Augmented Generation (RAG).

## Prerequisites
- Python 3.10 or 3.11 recommended
- Ollama installed locally
- Internet access for the first embedding-model download

## Windows installation

Open PowerShell in the DocuMind_AI folder:

    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

Install Ollama from https://ollama.com/

Then pull the language model:

    ollama pull qwen2.5:3b

DocuMind AI accepts PDF and TXT files up to **10 MB per document**. Each
document must also produce no more than **400 extracted text chunks**. These
are separate limits: a file can be below 10 MB but still be rejected if its
extracted text produces too many chunks. Documents that exceed either limit
are rejected before embeddings or database records are created.

After changing `.streamlit/config.toml`, restart Streamlit so the 10 MB upload
limit is applied:

    streamlit run app.py

Start the application:

    streamlit run app.py

## Demonstration
1. Open Document Management.
2. Upload the three TXT files from sample_docs.
3. Click Index uploaded documents.
4. Open AI Assistant and select the documents.
5. Ask about leave, incident reporting, and API limits.
6. Inspect the retrieved source passages.
7. Test document selection and deletion.

## Test questions
- How many paid annual leave days are available?
  Expected: 18 days per calendar year.
- How quickly must a security incident be reported?
  Expected: Within one hour of discovery.
- What is the API rate limit?
  Expected: 120 requests per minute per client application.
- What is the leave entitlement and API rate limit?
  Expected: Both facts with the relevant sources.
- What was company revenue in 2025?
  Expected: Insufficient information.

## Automated tests

    python -m unittest discover -s tests -v

## Limitations
- Scanned PDFs need an OCR extension.
- Answers and citations require human review.
- Retrieval distance thresholds need evaluation and tuning.
- Chat history is session-based, not permanently stored.
- Local single-user prototype; no authentication.
- Synthetic sample documents are for demonstrations only.
