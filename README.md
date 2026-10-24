# PDFQuery

A PDF question-answering app built with Python, LangChain, ChromaDB and Streamlit.

I built PDFQuery to work through the full RAG workflow: extracting PDF text, splitting it into chunks, retrieving relevant passages and generating answers with sources. Users can upload their own document and ask follow-up questions without repeating the subject each time.

## Features

- Upload and process one text-based PDF at a time, up to 20 MB.
- Ask questions about the document in English or Hindi/Hinglish.
- View retrieved passages with filenames and PDF page numbers.
- Use recent conversation history to resolve follow-up questions before retrieval.
- Show warnings when an answer has missing or unknown source labels.
- Clear the conversation or download it as a text file with source passages.
- Reset the conversation and active index when the uploaded document changes.

The app is instructed to answer from the retrieved PDF passages. When supporting information is unavailable, it should respond:

> I could not find the answer in the uploaded PDF.

This is an intended behavior, not a guarantee that every unsupported question will be handled correctly.

## How it works

1. Extract text from the uploaded PDF using PyPDFLoader.
2. Split the text into overlapping chunks.
3. Generate embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
4. Store the embeddings in ChromaDB.
5. Rewrite follow-up questions into standalone retrieval queries using recent chat history.
6. Retrieve relevant chunks and pass them to a Groq-hosted LLM.
7. Display the answer alongside the retrieved passages and check its source labels.

History helps interpret the question; the answer is generated from the current retrieved passages.

## Tech stack

| Component | Technology |
| --- | --- |
| Language | Python |
| Interface | Streamlit |
| RAG orchestration | LangChain |
| PDF extraction | PyPDFLoader |
| Text splitting | RecursiveCharacterTextSplitter |
| Embeddings | Hugging Face Sentence Transformers |
| Vector store | ChromaDB |
| LLM provider | Groq API |

The embedding model runs on the machine hosting the app. Answer generation and follow-up rewriting use the Groq API. FAISS is not used in this project.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/RishabhSuratiya02/PDFQuery.git
cd PDFQuery
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

The first run needs an internet connection to download the embedding model. Groq requests also require an internet connection and a valid API key.

### 5. Start the app

Run from the project root:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, upload a PDF and select **Process document**. Once processing finishes, enter a question in the chat box.

## Example questions

For a university syllabus PDF:

```text
What is the course code of Python Programming?
List the topics in Unit 1 of Computer Networks.
Now show Unit 2 of the same subject.
Compare the topics in those two units.
```

For a refusal check, ask for something you have confirmed is absent from the PDF, such as the name of your college's Python instructor.

## Checking answers

An answer appearing on screen does not prove it is correct. Expand **View retrieved PDF passages** and check that:

- Each citation points to the expected passage.
- The passage supports the claim in the answer.
- The page number matches the physical page order in the PDF viewer.

The citation checker validates source-label presence and existence. It does not verify whether a passage supports a claim, or whether every claim has a citation.

Useful regression checks include paraphrased questions, unsupported questions, follow-ups, clearing history and switching to an unrelated PDF. After switching documents, the app should not answer from the previous PDF.

## Limitations

- Scanned PDFs need OCR, which is not implemented.
- Retrieval can miss relevant passages, especially in tables or poorly extracted layouts.
- A summary uses retrieved chunks and may leave out parts of the document. Full-document summarization is not implemented.
- Follow-up memory is limited to recent messages in the current session; it is not persistent memory.
- LLM answers and citations still need verification.
- API rate limits and model availability can interrupt requests.
- The app handles one active PDF at a time.

## Data handling

PDF text is extracted and embedded on the app host. Retrieved passages are sent to Groq to generate answers. Recent chat messages are also sent to Groq when rewriting follow-up questions.

Do not upload confidential documents unless you are permitted to process them this way. Keep `.env`, API keys, local secrets, virtual environments, uploaded PDFs and generated vector-store data out of Git.

## Next improvements

- A repeatable evaluation set for retrieval, answer support and refusal behavior.
- Dedicated section and full-document summarization.
- Source-page preview inside the app.
- Index reuse and improved cleanup of unused collections.

## Author

**Rishabh Suratiya**

[GitHub](https://github.com/RishabhSuratiya02) · [LinkedIn](https://www.linkedin.com/in/rishabh-suratiya/)

