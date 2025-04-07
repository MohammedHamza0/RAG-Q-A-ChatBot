from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv(override=True)
import os


def PineCone(index_name, model_embedding, text_chunks):
    """
    Creates a new Pinecone index and uploads text chunks into it using the provided embedding model.

    Parameters:
    -----------
    index_name : str
        The name of the Pinecone index to create or use.
    
    model_embedding : HuggingFaceEmbeddings or compatible
        The embedding model to convert text chunks into vector embeddings.
    
    text_chunks : list
        A list of LangChain Document objects that represent the text data to be embedded and stored.

    Returns:
    --------
    docsearch : PineconeVectorStore
        A LangChain PineconeVectorStore object that allows performing similarity searches on the indexed data.
    """
    
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index only if it doesn't exist
    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(name=index_name,
                        metric="cosine",
                        dimension=768,
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))

    docsearch = PineconeVectorStore.from_documents(documents=text_chunks, 
                                                   index_name=index_name, 
                                                   embedding=model_embedding)
    
    return docsearch
