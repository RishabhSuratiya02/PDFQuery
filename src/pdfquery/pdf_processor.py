from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdfquery.config import chunk_size, chunk_overlap

def process_pdf(pdf_path: str | Path) -> list[Document]:
    path = Path(pdf_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Please upload a PDF file.")

    try:
        loader = PyPDFLoader(str(path))
        pages = loader.load()

    except Exception as exc:
        raise ValueError(
            "Could not read the PDF. It may be corrupted or password-protected."
        ) from exc

    readable_pages = []

    for page in pages:
        if not page.page_content.strip():
            continue

        page.metadata["filename"] = path.name

        page_index = page.metadata.get("page")
        if isinstance(page_index, int):
            page.metadata["page_number"] = page_index + 1

        readable_pages.append(page)

    if not readable_pages:
        raise ValueError(
            "No readable text was found in this PDF. "
            "If it is scanned, OCR is required."
    )

     #  Split pages into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(readable_pages)

    return [
        chunk for chunk in chunks
        if chunk.page_content.strip()
    ]