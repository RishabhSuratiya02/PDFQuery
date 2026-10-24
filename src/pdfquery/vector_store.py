# src/vector_store.py

from functools import lru_cache
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    embedding_model,
    retrieval_k,
    CHROMA_DIR,
    create_directories,
)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once per Python process."""

    return HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(chunks: list[Document]) -> Chroma:
    """Create a separate persistent collection for one PDF."""

    readable_chunks = [
        chunk for chunk in chunks
        if chunk.page_content.strip()
    ]

    if not readable_chunks:
        raise ValueError("No readable chunks available for indexing.")

    create_directories()

    collection_name = f"pdf_{uuid4().hex}"

    vector_store = Chroma.from_documents(
        documents=readable_chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
    )

    return vector_store


def create_retriever(vector_store: Chroma) -> BaseRetriever:
    """Create a retriever for the selected PDF collection."""

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retrieval_k},
    )