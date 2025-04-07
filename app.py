import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile

# Title
st.title("📄 PDF Text Extractor and Chunker")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    # Extraction
    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()

    st.subheader("✅ Extracted Text")
    for doc in documents:
        st.write(doc.page_content)

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
        is_separator_regex=False
    )
    chunks = splitter.split_documents(documents)

    st.subheader("🧩 Chunks")
    for i, chunk in enumerate(chunks):
        st.markdown(f"**Chunk {i+1}:**")
        st.write(chunk.page_content)

    # Cleanup temp file
    os.remove(temp_pdf_path)
