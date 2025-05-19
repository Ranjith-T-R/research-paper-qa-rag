import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
UPLOAD_FOLDER = "docs"
DB_FOLDER = "vector_store_db"
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

# Save uploaded PDF
def save_pdf(uploaded_file):
    path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

# Load and split PDF
def load_and_split_docs(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Create vector store from documents
def create_vector_store(docs, db_path):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}  # Fixes meta tensor error
    )
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(db_path)
    return db

def load_vector_store(db_path):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)


# Create QA chain with Ollama LLaMA 2
def get_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model="llama2")  # Run `ollama run llama2` beforehand
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# --- STREAMLIT APP ---
st.set_page_config(page_title="📚 RAG AI Assistant")
st.title("📚 AI Research Assistant using RAG")

# Sidebar
st.sidebar.header("Document Manager")

# Upload new PDF
uploaded_file = st.sidebar.file_uploader("Upload Research Papers", type=["pdf"])
if uploaded_file:
    saved_path = save_pdf(uploaded_file)
    st.sidebar.success(f" DOC Uploaded: {uploaded_file.name}")

# List available PDFs
# Refresh dropdown list after upload
if uploaded_file:
    st.experimental_rerun()

pdf_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".pdf")]
selected_pdf = st.sidebar.selectbox("Select a Paper to analyze", options=pdf_files)


if selected_pdf:
    pdf_path = os.path.join(UPLOAD_FOLDER, selected_pdf)
    db_path = os.path.join(DB_FOLDER, os.path.splitext(selected_pdf)[0])

    # Create or load vector DB
    if not os.path.exists(db_path):
        with st.spinner("Creating new Vector DB..."):
            docs = load_and_split_docs(pdf_path)
            vector_store = create_vector_store(docs, db_path)
    else:
        with st.spinner("Loading Vector DB..."):
            vector_store = load_vector_store(db_path)

    qa_chain = get_qa_chain(vector_store)

    st.success(f"Ready! Now asking questions about: **{selected_pdf}**")

    user_query = st.text_input(" Ask a question about the Research Documents")
    if user_query:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(user_query)
        st.write(" Answer...:", answer)
