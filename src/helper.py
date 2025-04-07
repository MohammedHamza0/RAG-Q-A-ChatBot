from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile
import os


def extract_text_from_pdf(uploaded_file):
    """
    Extracts text content from an uploaded PDF file using PyPDFLoader.

    Parameters:
    -----------
    uploaded_file : UploadedFile
        The uploaded PDF file from the Streamlit file uploader.

    Returns:
    --------
    documents : list
        A list of Document objects extracted from the PDF.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()

    os.remove(temp_pdf_path)  # Cleanup temp file after loading
    return documents


def chunk_text(documents):
    """
    Splits the extracted documents into smaller chunks using recursive character splitting.

    Parameters:
    -----------
    documents : list
        A list of Document objects.

    Returns:
    --------
    text_chunks : list
        A list of Document chunks with preserved structure.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
        is_separator_regex=False
    )
    text_chunks = splitter.split_documents(documents)
    return text_chunks


def download_embedding_model():
    """
    Downloads and initializes a Hugging Face embedding model.

    Returns:
    --------
    embedding_model : HuggingFaceEmbeddings
        An instance of the embedding model ready for use.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="ibm-granite/granite-embedding-125m-english")
    return embedding_model