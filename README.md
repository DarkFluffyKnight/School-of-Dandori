# School of Dandori — Course Portal

A Streamlit web app for browsing, filtering, and discovering whimsical adult education courses. Includes an AI assistant that uses Retrieval Augmented Generation (RAG) to recommend courses based on natural language prompts.

---

## Features

**Course Portal**
- Browse all courses in a paginated Discovery Gallery or a sortable Data Table
- Filter by location, category, instructor, skills developed, and price range
- Keyword search across course names and descriptions
- Sort by name, price, location, or instructor
- Save courses to a personal favourites list

**Dandori AI Assistant**
- Natural language course recommendations powered by RAG
- Semantic search over course embeddings via ChromaDB
- Conversational memory — the assistant rewrites follow-up questions into standalone queries using chat history
- Metadata filtering support (price, location, course type, etc.)

**Data Pipeline**
- Course details are extracted from PDF files using `pdfplumber`
- Extracted data is stored in `course_data.csv`
- Courses are embedded and stored in a persistent ChromaDB vector store

---

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── course_data.csv         # Extracted course data
├── all_pdfs/               # Full set of course PDFs (211 classes)
├── course_pdfs/            # Sample subset of PDFs
├── utils/
│   ├── parsepdf.py         # PDF extraction pipeline
│   ├── rag.py              # RAG utilities (embeddings, ChromaDB, LLM querying)
│   └── getters.py          # Data loading, cleaning, and filter helpers
└── chroma_db/              # Persistent ChromaDB vector store (auto-created)
```

---

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure environment variables**

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
ENDPOINT=https://openrouter.ai/api/v1
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=models/gemini-2.5-flash
EMBEDDING_MODEL=google/gemini-embedding-001
QUERY_MODEL=google/gemini-2.0-flash-001
CHROMA_DB_PATH=./chroma_db
```

**3. (Optional) Re-extract course data from PDFs**

If you want to regenerate `course_data.csv` from the PDFs:

```bash
python utils/parsepdf.py ./all_pdfs
```

**4. Run the app**

```bash
streamlit run app.py
```

On first run, if no ChromaDB collection is found, the app will automatically embed the course data and create one.

---

## How the RAG Pipeline Works

1. Course data from `course_data.csv` is chunked into text documents, one per course, containing all relevant fields.
2. Chunks are embedded using `google/gemini-embedding-001` via the OpenRouter API and stored in a persistent ChromaDB collection.
3. When a user sends a message, the assistant rewrites the query into a standalone search query using the conversation history.
4. The rewritten query is used to retrieve the most semantically similar courses from ChromaDB.
5. Retrieved course documents are injected as context into the LLM prompt alongside the original user question.
6. The LLM (`google/gemini-2.0-flash-001` by default) generates a response grounded in the retrieved context.

---

## PDF Extraction

`utils/parsepdf.py` processes PDFs with a fixed layout. For each file it extracts:

- Course name, ID, instructor, location, course type, and cost
- Learning objectives and provided materials (bullet point sections)
- Skills developed (parsed from a two-column boxed layout on page 2)
- Full course description

To process a folder of PDFs and overwrite `course_data.csv`:

```bash
python utils/parsepdf.py ./all_pdfs
```

---

## Models Used

| Purpose | Model |
|---|---|
| Embeddings | `google/gemini-embedding-001` (via OpenRouter) — `EMBEDDING_MODEL` |
| Query rewriting | `google/gemini-2.5-flash` (via OpenRouter) — `QUERY_MODEL` |
| Chat responses (Gemini SDK) | `models/gemini-2.5-flash` — `GEMINI_MODEL` |

All models are configurable via environment variables.
