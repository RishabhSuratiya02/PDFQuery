from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq

from pdfquery.config import (
    GROQ_API_KEY,
    llm_model,
    temperature,
    max_output_tokens,
    validate_config
)

from pdfquery.prompts import QA_PROMPT

Not_found = "I could not find the answer in the uploaded PDF."

def format_context(document:  list[Document])-> str:
    """Format retrieved chunks with source labels."""

    passages = []

    for index, document in enumerate(document, start=1):
        passages.append(
            f"[S{index}]\n{document.page_content}"
        )

    return "\n\n".join(passages)


def build_sources(documents: list[Document]) -> list[dict]:
    """Map source labels to actual PDF metadata and passages."""

    sources = []

    for index, document in enumerate(documents, start=1):
        sources.append(
            {
                "label": f"S{index}",
                "filename": document.metadata.get(
                    "filename", "Uploaded PDF"
                ),
                "page_number": document.metadata.get("page_number"),
                "text": document.page_content,
            }
        )

    return sources


class PDFQueryPipeline:
    def __init__(self, retriever: BaseRetriever):
        validate_config()

        self.retriever = retriever

        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=llm_model,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )

        # Runnable chain
        self.chain = QA_PROMPT | llm | StrOutputParser()

    def ask(self, question: str) -> dict:
        """Retrieve PDF passages and return an answer with sources."""

        question = question.strip()

        if not question:
            raise ValueError("Please enter a question.")

        # Retrieve chunks from PDF
        documents = self.retriever.invoke(question)

        documents = [
            document
            for document in documents
            if document.page_content.strip()
        ]

        
        if not documents:
            return {
                "answer": Not_found,
                "sources": [],
            }

        # Generate an answer from the retrieved context
        answer = self.chain.invoke(
            {
                "context": format_context(documents),
                "question": question,
            }
        )

        
        return {
            "answer": answer.strip(),
            "sources": build_sources(documents),
        }