# DocuMind AI — Upload Limit and Document Indexing Improvements

## Objective

Review the existing DocuMind AI Streamlit application and improve its document upload and indexing workflow.

**Do not rebuild the project. Modify the existing implementation.**

## Required Changes

### 1. Maximum File Upload Size

- Set the maximum file upload size to **10 MB per document**.
- Create or update `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 10
```

- Preserve any unrelated configuration.
- Ensure the upload interface displays 10 MB instead of 200 MB.

### 2. Backend File Validation

Update `index_document()` to validate file size before extracting text, generating embeddings, or inserting records.

Use:

```python
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
```

If a document exceeds the limit, show a clear error identifying the file.

### 3. Document Chunk Limit

The application currently limits documents to 400 chunks.

- Keep this limit for the prototype.
- Validate the chunk count before embedding generation.
- Explain clearly when a document exceeds the limit, even if its file size is below 10 MB.
- Never silently truncate the document.
- Ensure rejected documents do not leave partially indexed data.

### 4. Improve Upload Experience

- Disable the **Index uploaded documents** button when no files are selected.
- Clear the file uploader after indexing.
- Display temporary success and error notifications using `st.toast()`.
- Continue processing other files if one document fails.
- Show newly indexed documents under **My Documents**.

### 5. Preserve Existing Features

Do not remove or break:

- PDF and TXT uploads
- Multiple-document indexing
- Sentence Transformers embeddings
- ChromaDB vector storage
- Ollama and Qwen integration
- RAG question answering
- Source filename and page references
- Document deletion confirmation
- AI Assistant and My Documents navigation

Do not add sign-in/sign-up or expose advanced model settings.

### 6. Testing

Verify:

1. Documents below 10 MB are accepted.
2. Documents above 10 MB are rejected.
3. Boundary values are handled correctly.
4. Files producing more than 400 chunks are rejected clearly.
5. Failed uploads do not create incomplete database records.
6. Upload fields reset after processing.
7. Existing RAG functionality continues working.

Run available tests and report the actual results. Clearly identify any tests that could not be completed.

### 7. Documentation

Update `README.md` with the upload limits and restart instructions.

Explain that file size and extracted-text length are separate constraints.

## Expected Deliverables

- Updated existing project files.
- Consistent upload-size limits.
- Improved upload validation.
- Updated documentation.
- Test results.
- Summary of modifications.

**Inspect the existing implementation before making changes. Preserve all working features.**