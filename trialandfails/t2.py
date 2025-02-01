import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.title("LawAssist - Indian Penal Code RAG Chatbot")

# Initialize the language model
llm = ChatGroq(model="gemma-7b-it")

# Define the prompt template for answering questions
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Load the text file containing the Indian Penal Code
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to create vector embeddings and prepare the vector store
def vector_embedding():
    if "vectors" not in st.session_state:
        # Path to the text file
        text_file_path = r"C:\Users\Shriya Deshpande\OneDrive\Desktop\gemma chatbot\input1.txt"

        if not os.path.exists(text_file_path):
            st.error("The specified text file does not exist.")
            return

        # Load the content of the text file
        text_content = load_text_file(text_file_path)
        

        if not text_content:
            st.error("The text file is empty.")
            return

        # Convert the text content into a list of documents
        documents = [{'page_content': text_content}]

        # Create document chunks from the text content
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)

    if not final_documents:
            st.error("Failed to split the document into chunks.")
            return

        # Initialize embeddings
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Create a FAISS vector store from the document chunks
    st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
    st.write("Vector Store DB Is Ready")

# Load and embed the document when the button is clicked
if st.button("Load and Embed Document"):
    vector_embedding()

import time

# Input field for the user to ask questions
prompt1 = st.text_input("Enter Your Question From Documents")

if prompt1:
    # Create a document chain and retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Measure the time taken for the response
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write("Response time: ", time.process_time() - start)

    # Display the answer
    st.write(response['answer'])

    # Display the relevant document chunks
    with st.expander("Document Similarity Search"):
        if "context" in response:
            for i, doc in enumerate(response["context"]):
                st.write(doc['page_content'])
                st.write("--------------------------------")
        else:
            st.write("No context found.")
