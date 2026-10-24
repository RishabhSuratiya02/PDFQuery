# app.py

import hashlib
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st
from groq import AuthenticationError, RateLimitError


# Make the src/pdfquery package available when running from project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pdfquery.config import validate_config
from pdfquery.pdf_processor import process_pdf
from pdfquery.rag_pipeline import PDFQueryPipeline
from pdfquery.vector_store import create_vector_store, create_retriever


logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="PDFQuery | Chat with your PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Appearance ----------

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1120;
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] {
        background: #111a2d;
        border-right: 1px solid #243047;
    }

    .block-container {
        max-width: 1080px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    .brand {
        color: #818cf8;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: clamp(2rem, 5vw, 3.4rem);
        font-weight: 750;
        line-height: 1.15;
        letter-spacing: -0.045em;
        margin-bottom: 0.7rem;
        color: #f8fafc;
    }

    .hero-copy {
        color: #94a3b8;
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 680px;
        margin-bottom: 1.8rem;
    }

    .feature-card {
        background: #121c30;
        border: 1px solid #26334b;
        border-radius: 16px;
        padding: 1.3rem;
        min-height: 150px;
        margin-bottom: 1rem;
    }

    .feature-card h4 {
        color: #e2e8f0;
        margin: 0 0 0.6rem;
        font-size: 1rem;
    }

    .feature-card p {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0;
    }

    [data-testid="stChatMessage"] {
        background: #121c30;
        border: 1px solid #243047;
        border-radius: 16px;
        margin-bottom: 1rem;
    }

    .stButton > button[kind="primary"] {
        background: #6366f1;
        border: 1px solid #818cf8;
        border-radius: 10px;
        color: white;
    }

    .stButton > button[kind="primary"]:hover {
        background: #4f46e5;
        border-color: #a5b4fc;
        color: white;
    }

    [data-testid="stMetricValue"] {
        color: #a5b4fc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Session state ----------

defaults = {
    "pipeline": None,
    "vector_store": None,
    "document_key": None,
    "document_name": None,
    "messages": [],
    "chunk_count": 0,
    "page_count": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def delete_collection(store):
    """Delete only the collection created for this session."""
    if store is not None:
        try:
            store.delete_collection()
        except Exception:
            logger.exception("Could not clean up PDF collection.")


def reset_document():
    """Disconnect the previous PDF and clear its conversation."""
    old_store = st.session_state.vector_store

    st.session_state.pipeline = None
    st.session_state.vector_store = None
    st.session_state.document_key = None
    st.session_state.document_name = None
    st.session_state.messages = []
    st.session_state.chunk_count = 0
    st.session_state.page_count = 0

    delete_collection(old_store)


def render_citation_warning(check):
    if not check:
        return

    if check.get("invalid_labels"):
        st.warning(
            "Some source labels could not be matched to "
            "retrieved passages. Verify this answer."
        )
    elif check.get("missing_citations"):
        st.warning(
            "This response has no source labels. "
            "Verify it against the retrieved PDF passages."
        )

def render_sources(sources):
    if not sources:
        return

    with st.expander("View retrieved PDF passages", expanded=False):
        st.caption(
            "These passages were retrieved for this question. "
            "Source labels in the answer refer to this list."
        )

        for source in sources:
            page = source.get("page_number")
            page_text = f"Page {page}" if page is not None else "Page unknown"

            st.markdown(f"**[{source['label']}] · {page_text}**")
            st.caption(source.get("filename", "Uploaded PDF"))

            # Render document text without interpreting it as HTML
            st.text(source.get("text", ""))
            st.divider()


def export_chat():
    lines = [
        "PDFQuery conversation",
        f"Document: {st.session_state.document_name}",
        "",
    ]

    for message in st.session_state.messages:
        lines.append(f"{message['role'].upper()}:")
        lines.append(message["content"])

        for source in message.get("sources", []):
            lines.append(
                f"[{source['label']}] "
                f"{source.get('filename', 'Uploaded PDF')} — "
                f"Page {source.get('page_number', 'unknown')}"
            )
            lines.append(source.get("text", ""))

        lines.append("")

    return "\n".join(lines)


# ---------- Sidebar ----------

with st.sidebar:
    st.markdown("## 📄 PDFQuery")
    st.caption("Your document. Your questions.")
    st.divider()

    st.markdown("### Add your document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        on_change=reset_document,
        help="Upload a text-based PDF up to 20 MB.",
    )

    st.caption(
        "Relevant PDF passages are sent to Groq to generate answers."
    )

    oversized = (
        uploaded_file is not None
        and uploaded_file.size > 20 * 1024 * 1024
    )

    if oversized:
        st.error("Please choose a PDF smaller than 20 MB.")

    process_clicked = st.button(
        "Process document",
        type="primary",
        use_container_width=True,
        disabled=(
            uploaded_file is None
            or oversized
            or st.session_state.pipeline is not None
        ),
    )

    if process_clicked:
        new_store = None

        try:
            validate_config()
            pdf_bytes = uploaded_file.getvalue()

            if not pdf_bytes:
                raise ValueError("The uploaded file is empty.")

            document_key = hashlib.sha256(pdf_bytes).hexdigest()

            with st.status("Preparing your PDF…", expanded=True) as status:
                st.write("Extracting text and splitting pages…")

                # Temporary PDF is removed after parsing
                with TemporaryDirectory(prefix="pdfquery_") as folder:
                    pdf_path = Path(folder) / "document.pdf"
                    pdf_path.write_bytes(pdf_bytes)
                    chunks = process_pdf(pdf_path)

                # Preserve the original upload name in source metadata
                for chunk in chunks:
                    chunk.metadata["filename"] = uploaded_file.name

                st.write("Creating embeddings and the search index…")
                new_store = create_vector_store(chunks)

                st.write("Connecting the question-answering pipeline…")
                retriever = create_retriever(new_store)
                pipeline = PDFQueryPipeline(retriever)

                st.session_state.vector_store = new_store
                st.session_state.pipeline = pipeline
                st.session_state.document_key = document_key
                st.session_state.document_name = uploaded_file.name
                st.session_state.chunk_count = len(chunks)
                st.session_state.page_count = len(
                    {
                        chunk.metadata["page_number"]
                        for chunk in chunks
                        if "page_number" in chunk.metadata
                    }
                )

                status.update(
                    label="Your PDF is ready",
                    state="complete",
                    expanded=False,
                )

        except Exception:
            logger.exception("PDF processing failed.")
            delete_collection(new_store)

            st.error(
                "Could not prepare this PDF. Check that it contains "
                "readable text, is not password-protected, and your "
                "API key and embedding download are available. "
                "Technical details are in the terminal."
            )

    st.divider()

    if st.session_state.pipeline is not None:
        st.success("Document ready")
        st.write(st.session_state.document_name)

        left, right = st.columns(2)
        left.metric("Text pages", st.session_state.page_count)
        right.metric("Chunks", st.session_state.chunk_count)

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.session_state.messages:
            st.download_button(
                "Download conversation",
                data=export_chat(),
                file_name="pdfquery_conversation.txt",
                mime="text/plain",
                use_container_width=True,
            )
    else:
        st.info("Upload and process a PDF to start.")

    st.divider()
    st.caption(
        "Text-based PDFs supported. Scanned pages need OCR. "
        "Verify important answers against the source passages."
    )


# ---------- Main area ----------

st.markdown(
    '<div class="brand">DOCUMENT QUESTION ANSWERING</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-title">Your PDF, a conversation away.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-copy">
        Ask questions about your document, explore its details,
        and inspect the passages behind each response.
    </div>
    """,
    unsafe_allow_html=True,
)

ready = st.session_state.pipeline is not None

if not ready:
    columns = st.columns(3)

    cards = [
        (
            "01 · Upload",
            "Choose a text-based PDF using the sidebar.",
        ),
        (
            "02 · Ask",
            "Ask a specific question about its contents.",
        ),
        (
            "03 · Check",
            "Inspect the retrieved passages and source pages.",
        ),
    ]

    for column, (title, description) in zip(columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="feature-card">
                    <h4>{title}</h4>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.info("Start by uploading a PDF in the sidebar.")

else:
    st.caption(f"Active document: {st.session_state.document_name}")
    st.caption(
        "Ask complete questions. Follow-up memory is not enabled yet."
    )

    if not st.session_state.messages:
        st.markdown("#### Start with a question")
        st.caption(
            'For example: “What does the document say about eligibility?”'
        )


# ---------- Chat history ----------

for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "📄"

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_citation_warning(message.get("citation_check"))
            render_sources(message.get("sources", []))


# ---------- Question answering ----------

question = st.chat_input(
    "Ask a question about your PDF…",
    disabled=not ready,
    max_chars=4000,
)

if question and question.strip():
    question = question.strip()

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📄"):
        try:
            with st.spinner("Searching your PDF…"):
                result = st.session_state.pipeline.ask(
                question,
                history=st.session_state.messages[:-1],
            )

            st.markdown(result["answer"])
            render_citation_warning(result.get("citation_check"))
            render_sources(result.get("sources", []))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "citation_check": result.get("citation_check"),

                }
            )

        except RateLimitError:
            st.warning(
                "The model's usage limit was reached. "
                "Wait a little, then submit your question again."
            )

        except AuthenticationError:
            st.error(
                "Groq authentication failed. Check GROQ_API_KEY "
                "in your .env file and restart the app."
            )

        except Exception:
            logger.exception("Question answering failed.")
            st.error(
                "Could not generate an answer. Please try again. "
                "Technical details are in the terminal."
            )