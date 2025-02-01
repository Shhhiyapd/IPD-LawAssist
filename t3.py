#real one shown for 1st presentation
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS #vectordb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings #vectorembeddings
from langchain.docstore.document import Document

from dotenv import load_dotenv

load_dotenv()

#load api keys
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("LawAssist")

llm = ChatGroq(model="gemma2-9b-it")

# Load the text file containing the Indian Penal Code
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


prompt=ChatPromptTemplate.from_template(
"""
Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
provided context just say, "answer is not available in the context", don't provide the wrong answer
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)


def vector_embedding():
    if "vectors" not in st.session_state:
        text_file_path = r"C:\Users\Shriya Deshpande\OneDrive\Desktop\gemma chatbot\input1.txt"

        # Load the text file
        st.session_state.docs = load_text_file(text_file_path)
        
        # Create document objects from the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(st.session_state.docs)
        documents = [Document(page_content=chunk) for chunk in chunks]

        # Set up embeddings and vector store
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vectors = FAISS.from_documents(documents, st.session_state.embeddings)


prompt1=st.text_input("Enter Your Question From Documents")


if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time



if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])
    print(response)

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")




