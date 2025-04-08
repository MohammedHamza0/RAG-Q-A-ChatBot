import streamlit as st
st.set_page_config(page_title="Q&A Bot", page_icon="🧾")

import os
from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt
from src.helper import extract_text_from_pdf, chunk_text, download_embedding_model
from store_index import PineCone_db
from tqdm.auto import tqdm
import time
from langchain_core.documents import Document




GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)
prompt = ChatPromptTemplate.from_template(system_prompt)

question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# Cache the model embedding to avoid downloading it every time
@st.cache_resource
def get_embedding_model():
    return download_embedding_model()

# Initialize the model embedding
model_embedding = get_embedding_model()


def main():
     # Custom CSS styling
     st.markdown("""
        <style>
        .chat-container {
            width: 100%;
            max-width: 700px;
            margin: 0 auto;
            background-color: #131722;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .chat-message {
            margin-bottom: 12px;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chat-message.user {
            background-color: #0a5e2a;
            text-align: right;
            flex-direction: row-reverse;
            color: #e8ebf1;
        }
        .chat-message.bot {
            background-color: #e1f7d5;
            text-align: left;
            flex-direction: row;
            color: #131722;
        }
        .chat-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .header {
            text-align: center;
            padding: 10px;
            background-color: #1E90FF;
            color: white;
            border-radius: 8px;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
            background-color: #1d2330;
            color: #e8ebf1;
        }
        body {
            color: #e8ebf1;
            background-color: #131722;
        }
        </style>
    """, unsafe_allow_html=True)

     # Avatars
     bot_avatar = "https://img.freepik.com/free-vector/graident-ai-robot-vectorart_78370-4114.jpg"
     user_avatar = "https://cdn-icons-png.freepik.com/512/6596/6596121.png"

     st.markdown("<h2 class='header'>Q&A Bot</h2>", unsafe_allow_html=True)
     st.write("Ask me anything related to your file")
     
     uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
     if uploaded_file is not None:
          extracted_data = extract_text_from_pdf(uploaded_file)
          chunks = chunk_text(extracted_data)
          st.sidebar.write(f"Number of chunks: {len(chunks)}")
          button = st.sidebar.button("Process")
          if button:
               with st.sidebar.status("Processing the file...", expanded=True) as status:
                    progress_bar = st.progress(0)
                    status.write("Initializing Pinecone...")
                    # Create a placeholder for the progress text
                    progress_text = st.empty()
                    
                    # Initialize Pinecone with chunks 
                    docsearch = PineCone_db(index_name="q-a", model_embedding=model_embedding, text_chunks=chunks)
                    
                    # Update progress for each chunk
                    for i, chunk in enumerate(chunks):
                        progress = (i + 1) / len(chunks)
                        progress_bar.progress(progress)
                        progress_text.text(f"Processing chunk {i+1} of {len(chunks)}")
                        time.sleep(0.01)  # Small delay to show progress
                    
                    status.update(label="Processing complete!", state="complete", expanded=False)
                    st.sidebar.success(f"Successfully processed {len(chunks)} chunks!")
     
     
if __name__ == "__main__":
     main()
