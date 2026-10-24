import re

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq

from .config import (
    GROQ_API_KEY,
    llm_model,
    temperature,
    max_output_tokens,
    validate_config,
)
from .prompts import QA_PROMPT, REWRITE_PROMPT


NOT_FOUND = "I could not find the answer in the uploaded PDF."

CITATION_PATTERN = re.compile(
    r"[\[【]\s*S(\d+)\s*[\]】]"
)

# Latest six messages; individual messages are also bounded.
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 1500


def prepare_history(history: list[dict]) -> list:
    messages = []

    for message in history[-MAX_HISTORY_MESSAGES:]:
        role = message.get("role")
        content = str(message.get("content", ""))

        # Old citation labels belong to old retrieval results.
        content = CITATION_PATTERN.sub("", content)
        content = content[:MAX_HISTORY_CHARS]

        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    return messages


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[S{index}]\n{document.page_content}"
        for index, document in enumerate(documents, start=1)
    )


def build_sources(documents: list[Document]) -> list[dict]:
    return [
        {
            "label": f"S{index}",
            "filename": document.metadata.get(
                "filename", "Uploaded PDF"
            ),
            "page_number": document.metadata.get("page_number"),
            "text": document.page_content,
        }
        for index, document in enumerate(documents, start=1)
    ]


def check_citations(answer: str, sources: list[dict]) -> dict:
    """Check label existence, not whether claims are supported."""

    used_labels = {
        f"S{number}"
        for number in CITATION_PATTERN.findall(answer)
    }
    available_labels = {source["label"] for source in sources}

    invalid_labels = sorted(used_labels - available_labels)

    missing_citations = (
        answer.strip() != NOT_FOUND
        and not used_labels
    )

    return {
        "invalid_labels": invalid_labels,
        "missing_citations": missing_citations,
        "valid": not invalid_labels and not missing_citations,
    }


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

        self.rewrite_chain = (
            REWRITE_PROMPT | llm | StrOutputParser()
        )

        self.answer_chain = (
            QA_PROMPT | llm | StrOutputParser()
        )

    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
    ) -> dict:
        question = question.strip()

        if not question:
            raise ValueError("Please enter a question.")

        messages = prepare_history(history or [])

        # Resolve follow-up references before searching the PDF.
        standalone_question = question

        if messages:
            rewritten = self.rewrite_chain.invoke(
                {
                    "history": messages,
                    "question": question,
                }
            ).strip()

            if rewritten:
                standalone_question = rewritten

        documents = self.retriever.invoke(standalone_question)

        documents = [
            document
            for document in documents
            if document.page_content.strip()
        ]

        if not documents:
            return {
                "answer": NOT_FOUND,
                "sources": [],
                "standalone_question": standalone_question,
                "citation_check": {
                    "invalid_labels": [],
                    "missing_citations": False,
                    "valid": True,
                },
            }

        answer = self.answer_chain.invoke(
            {
                "question": question,
                "standalone_question": standalone_question,
                "context": format_context(documents),
            }
        ).strip()

        if not answer:
            raise RuntimeError("The model returned an empty answer.")

        sources = build_sources(documents)

        return {
            "answer": answer,
            "sources": sources,
            "standalone_question": standalone_question,
            "citation_check": check_citations(answer, sources),
        }