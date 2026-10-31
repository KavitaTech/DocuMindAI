import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

import chromadb
import fitz
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

from rag_core import chunk_text, build_context, SYSTEM_PROMPT


BASE = Path(__file__).resolve().parent
DATA = BASE / ".documind_data"
DATA.mkdir(exist_ok=True)

DB = DATA / "documents.sqlite3"
VECTOR_PATH = DATA / "chromadb"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_CHUNKS = 400

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)


def db_connection():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db_connection() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                indexed_at TEXT NOT NULL
            )
        """)


@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(VECTOR_PATH))
    return client.get_or_create_collection(
        name="documind_documents",
        metadata={"hnsw:space": "cosine"}
    )


@st.dialog("Delete document?")
def confirm_delete(doc_id, filename):

    st.warning(
        f"Are you sure you want to delete **{filename}**?"
    )

    st.write(
        "All indexed text chunks and embeddings for this "
        "document will be removed from DocuMind AI. "
    )

    cancel_col, delete_col = st.columns(2)

    with cancel_col:
        if st.button(
            "Cancel",
            use_container_width=True
        ):
            st.rerun()

    with delete_col:
        if st.button(
            "Delete document",
            type="primary",
            use_container_width=True
        ):
            try:
                delete_document(doc_id)

                # Save notification for the next rerun
                st.session_state.document_flash = [{
                    "message": f"{filename}: Deleted successfully",
                    "icon": "✅"
                }]

                st.rerun()

            except Exception as error:
                st.error(
                    f"Unable to delete document: {error}"
                )



def list_documents():
    with db_connection() as con:
        result = con.execute(
            "SELECT * FROM documents ORDER BY indexed_at DESC"
        ).fetchall()
    return [dict(item) for item in result]


def extract_pages(uploaded_file):
    raw = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".txt":
        return [(1, raw.decode("utf-8-sig"))]

    if suffix != ".pdf":
        raise ValueError("Only PDF and TXT documents are supported.")

    pdf = fitz.open(stream=raw, filetype="pdf")
    pages = []

    try:
        for page_number, page in enumerate(pdf, start=1):
            content = page.get_text("text").strip()
            if content:
                pages.append((page_number, content))
    finally:
        pdf.close()

    return pages


def index_document(uploaded_file):
    raw = uploaded_file.getvalue()

    if len(raw) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"{uploaded_file.name} exceeds the {MAX_FILE_SIZE_MB} MB "
            "per-document upload limit."
        )

    doc_id = hashlib.sha256(raw).hexdigest()[:24]

    with db_connection() as con:
        exists = con.execute(
            "SELECT 1 FROM documents WHERE doc_id = ?",
            (doc_id,)
        ).fetchone()

    if exists:
        return "Already indexed"

    pages = extract_pages(uploaded_file)
    chunks = []
    metadata = []

    for page_number, text in pages:
        for chunk in chunk_text(text):
            chunks.append(chunk)
            metadata.append({
                "doc_id": doc_id,
                "filename": uploaded_file.name,
                "page": page_number
            })

    if not chunks:
        raise ValueError(
            "No extractable text found. Scanned PDFs need OCR, "
            "which is not enabled in this version."
        )

    if len(chunks) > MAX_CHUNKS:
        raise ValueError(
            f"{uploaded_file.name} produces {len(chunks)} chunks, which "
            f"exceeds the {MAX_CHUNKS}-chunk prototype limit. The document "
            "was not indexed."
        )

    model = load_model()

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    collection = get_collection()
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

    try:
        for offset in range(0, len(chunks), 100):
            end = offset + 100
            collection.add(
                ids=ids[offset:end],
                documents=chunks[offset:end],
                metadatas=metadata[offset:end],
                embeddings=embeddings[offset:end]
            )

        with db_connection() as con:
            con.execute(
                """
                INSERT INTO documents
                (doc_id, filename, chunk_count, indexed_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    doc_id,
                    uploaded_file.name,
                    len(chunks),
                    datetime.now().isoformat(timespec="seconds")
                )
            )

    except Exception:
        collection.delete(where={"doc_id": doc_id})
        raise

    return f"Indexed {len(chunks)} chunks"


def delete_document(doc_id):
    get_collection().delete(where={"doc_id": doc_id})

    with db_connection() as con:
        con.execute(
            "DELETE FROM documents WHERE doc_id = ?",
            (doc_id,)
        )


def retrieve(question, selected_ids, top_k, max_distance):
    collection = get_collection()

    if collection.count() == 0 or not selected_ids:
        return []

    embedding = load_model().encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    if len(selected_ids) == 1:
        where = {"doc_id": selected_ids[0]}
    else:
        where = {"doc_id": {"$in": selected_ids}}

    results = collection.query(
        query_embeddings=embedding,
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    matches = []

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        if distance <= max_distance:
            matches.append({
                "text": text,
                "metadata": metadata,
                "distance": distance
            })

    return matches


def ask_ollama(question, matches, model_name):
    context = build_context(matches)

    user_message = (
        f"DOCUMENT EXCERPTS:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer only from the excerpts and include source citations."
    )

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "options": {"temperature": 0.1}
        },
        timeout=180
    )

    response.raise_for_status()
    return response.json()["message"]["content"]



# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

# Internal RAG settings — not displayed to users
model_name = "qwen2.5:3b"
top_k = 5
max_distance = 0.85

init_db()

# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("📚 DocuMind AI")
st.caption(
    "Intelligent Document Analysis & Question Answering using RAG"
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📚 DocuMind AI")
st.sidebar.caption("Your intelligent document assistant")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "💬 AI Assistant",
        "📄 My Documents"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.caption("Powered by RAG")

# ============================================================
# SESSION STATE
# ============================================================

if "upload_version" not in st.session_state:
    st.session_state.upload_version = 0

if "upload_results" not in st.session_state:
    st.session_state.upload_results = []

if "history" not in st.session_state:
    st.session_state.history = []

# Get indexed documents
documents = list_documents()

# ============================================================
# MY DOCUMENTS PAGE
# ============================================================


if page == "📄 My Documents":

    st.subheader("Upload and index documents")

    # -----------------------------------------
    # TEMPORARY FLASH NOTIFICATIONS
    # -----------------------------------------

    # Show messages once, then remove them
    flash_messages = st.session_state.pop(
        "document_flash", []
    )

    for flash in flash_messages:
        st.toast(
            flash["message"],
            icon=flash["icon"]
        )

    # -----------------------------------------
    # DOCUMENT UPLOAD
    # -----------------------------------------

    uploads = st.file_uploader(
        "Upload PDF or TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key=f"document_upload_{st.session_state.upload_version}"
    )
    st.caption(
        f"Maximum {MAX_FILE_SIZE_MB} MB per document. "
        "Documents are also limited to 400 extracted text chunks."
    )

    index_clicked = st.button(
        "Index uploaded documents",
        type="primary",
        disabled=not bool(uploads)
    )

    # -----------------------------------------
    # INDEX DOCUMENTS
    # -----------------------------------------

    if index_clicked and uploads:

        flash_messages = []

        for uploaded_file in uploads:

            try:
                with st.spinner(
                    f"Indexing {uploaded_file.name}..."
                ):
                    message = index_document(uploaded_file)

                flash_messages.append({
                    "message": f"{uploaded_file.name}: {message}",
                    "icon": "✅" if message != "Already indexed" else "ℹ️"
                })

            except Exception as error:

                flash_messages.append({
                    "message": f"{uploaded_file.name}: {error}",
                    "icon": "❌"
                })

        # Store notifications for display after rerun
        st.session_state.document_flash = flash_messages

        # Reset the file uploader
        st.session_state.upload_version += 1

        st.rerun()

    # -----------------------------------------
    # INDEXED DOCUMENTS
    # -----------------------------------------

    st.divider()
    st.subheader("Indexed documents")

    documents = list_documents()

    if not documents:

        st.info("No documents indexed yet.")

    else:

        st.caption(
            f"{len(documents)} document(s) available"
        )

        for doc in documents:

            left, right = st.columns([5, 1])

            with left:
                st.write(
                    f"📄 **{doc['filename']}**"
                )

                st.caption(
                    f"{doc['chunk_count']} indexed chunks"
                )

            with right:

                if st.button(
                    "Delete",
                    key=f"delete_{doc['doc_id']}"
                ):

                    confirm_delete(
                        doc["doc_id"],
                        doc["filename"]
                    )

# ============================================================
# AI ASSISTANT PAGE
# ============================================================

elif page == "💬 AI Assistant":

    st.subheader("Ask questions about your documents")

    if not documents:

        st.info(
            "No documents available. "
            "Go to My Documents to upload and index documents."
        )

    else:

        # Map document IDs to document names
        names = {
            doc["doc_id"]: doc["filename"]
            for doc in documents
        }

        # Select documents for retrieval
        selected = st.multiselect(
            "Select documents to search",
            options=list(names.keys()),
            default=list(names.keys()),
            format_func=lambda doc_id: names[doc_id]
        )

        # ----------------------------------------------------
        # CHAT HISTORY
        # ----------------------------------------------------

        if st.button("Clear chat"):

            st.session_state.history = []
            st.rerun()

        for message in st.session_state.history:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # ----------------------------------------------------
        # USER QUESTION
        # ----------------------------------------------------

        question = st.chat_input(
            "Ask something about your documents..."
        )

        if question:

            st.session_state.history.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):

                # Reset retrieved matches for every question
                matches = []

                if not selected:

                    answer = (
                        "Please select at least one document."
                    )

                else:

                    with st.spinner(
                        "Searching documents and generating answer..."
                    ):

                        try:

                            # Retrieve relevant passages
                            matches = retrieve(
                                question,
                                selected,
                                top_k,
                                max_distance
                            )

                            if not matches:

                                answer = (
                                    "I could not find sufficient "
                                    "information in the uploaded documents."
                                )

                            else:

                                # Generate answer using Ollama
                                answer = ask_ollama(
                                    question,
                                    matches,
                                    model_name
                                )

                        except requests.RequestException as error:

                            answer = (
                                "Could not connect to Ollama. "
                                "Make sure Ollama is running and "
                                "the Qwen model is installed.\n\n"
                                f"Details: {error}"
                            )

                            matches = []

                        except Exception as error:

                            answer = (
                                f"Unable to process your question: {error}"
                            )

                            matches = []

                # Display AI answer
                st.markdown(answer)

                # ------------------------------------------------
                # SOURCE REFERENCES
                # ------------------------------------------------

                if matches:

                    with st.expander(
                        "📖 Retrieved source passages"
                    ):

                        for index, match in enumerate(
                            matches,
                            start=1
                        ):

                            metadata = match["metadata"]

                            st.markdown(
                                f"**[S{index}] "
                                f"{metadata['filename']} — "
                                f"Page {metadata['page']}**"
                            )

                            st.caption(
                                "Cosine distance: "
                                f"{match['distance']:.3f}"
                            )

                            st.write(match["text"])

            # Save assistant answer to chat history
            st.session_state.history.append({
                "role": "assistant",
                "content": answer
            })
