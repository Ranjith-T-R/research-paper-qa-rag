# Research Paper Question Answering Using Retrieval‑Augmented Generation (RAG)

 AI Research Assistant using RAG:

A Streamlit‑powered Retrieval‑Augmented Generation (RAG) assistant that lets you upload PDFs, build a FAISS vector index over text chunks (via HuggingFace embeddings), and query them using a local LLaMA 2 model served through Ollama.

---

## 🚀 Features

- **PDF Ingestion**  
  Upload research papers (PDF) directly through Streamlit.  
- **Chunking & Embeddings**  
  Extract text with `PyPDFLoader`, split into overlapping chunks via `CharacterTextSplitter`, and embed using `sentence-transformers/all-MiniLM-L6-v2`.  
- **Vector Indexing**  
  Build/search a FAISS index for fast nearest‑neighbor lookup. Saved locally for instant reloads.  
- **Local LLM QA**  
  Query your documents with a local LLaMA 2 model (`ollama`) for fast, private inference.  
- **Session‑state Caching**  
  Avoids redundant re-indexing or re-loading on every UI interaction.

---
## 📦 Installation

1. **Clone this repo**  
   ```bash
   git clone https://github.com/<your‑username>/ai-research-assistant-rag.git
   cd ai-research-assistant-rag
2. **Create & activate a virtual environment (recommended)**
    ```bash
   python3 -m venv .venv
   source .venv/bin/activate
 
3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
4. **Install & run Ollama**
   Follow Ollama’s quickstart to install.
   Then pull/run LLaMA 2 locally:
    ```bash
   ollama pull llama2
   ollama run llama2
 ---
## ⚙️ Usage

1. **Launch the Streamlit app**  
   ```bash
   streamlit run app.py
2. **Upload a PDF**
In the sidebar, click Upload Research Paper and choose any .pdf.

3. **Wait for vector DB**
The app will split, embed, and index your document (takes ~10–30 s).

4. **Ask questions**
Type any query in the input box and hit Enter.
– It retrieves the top 3 relevant chunks, sends them to LLaMA 2, and displays the answer.
---
## 🛠 Troubleshooting

1. **“meta tensor” error**
Ensure embeddings run on CPU by passing model_kwargs={"device": "cpu"}.

2. **Ollama connection issues**
Make sure ollama run llama2 is active before querying.

3. **Re‑upload looping**
Add a simple session‑state guard around st.experimental_rerun()
---
 
## 📁 Project Structure

```bash
├── app.py               # Streamlit entrypoint
├── requirements.txt     # Python dependencies
├── docs/                # Uploaded PDFs
│   └── example.pdf      # Sample research paper
├── vector_store_db/     # FAISS index folders
│   └── example/         # Index for example.pdf
└── .env                 # Optional environment variables
---
