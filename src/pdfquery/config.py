import os

from dotenv import load_dotenv

from pathlib import Path

# Project root: pdfquery/
BASE_DIR = Path(__file__).resolve().parents[2]

# Load API keys from the project's .env file
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#LLM model Integrate

llm_model = "openai/gpt-oss-20b"
temperature = 0
max_output_tokens = 800

#create Embedding
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

chunk_size = 1000
chunk_overlap = 200

# Retrieval settings
retrieval_k = 4


# Storage paths
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CHROMA_DIR = BASE_DIR / "data" / "chroma"


def validate_config() -> None:
    """Check required configuration before processing a PDF."""
    if not GROQ_API_KEY or not GROQ_API_KEY.strip():
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )


def create_directories() -> None:
    """Create storage folders when the application starts."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


