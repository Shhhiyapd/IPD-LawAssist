import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document

from dotenv import load_dotenv

load_dotenv()

# Load API keys
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("LawAssist")

llm = ChatGroq(model="gemma2-9b-it")

# Define paths for text file and PDF folder
textfile_path = r"C:\Users\Shriya Deshpande\OneDrive\Desktop\gemma chatbot\input1.txt"
pdffolder_path = r"C:\Users\Shriya Deshpande\OneDrive\Desktop\gemma chatbot\pdffiles"

# Load text file containing legal text
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_pdf_folder(folder_path):
    loader = PyPDFDirectoryLoader(folder_path)
    pdf_documents = loader.load()
    return pdf_documents

prompt = ChatPromptTemplate.from_template(
"""
Answer the following question in a detailed and accurate manner based on the provided legal context. Please follow these guidelines:

1. **Answer Structure**:
    - **Summary of Relevant Law**: Provide a brief overview of the most pertinent legal section(s).
    - **Detailed Explanation**: Elaborate on the law's application to the question, with a step-by-step breakdown.
    - **Relevant Case Laws or Precedents**: If applicable, reference notable cases that support the interpretation of the law.
    - **Suggestions for Further Reading**: Provide references for any additional sections or legal texts that the user may explore for more context.

2. **Multiple Perspectives**:
    - If the question involves multiple interpretations, outline each possible interpretation and clarify your reasoning for each.
    - If applicable, include the potential consequences or outcomes based on different interpretations.

3. **Handling Ambiguity**:
    - If the question is unclear or too broad, identify the key legal terms or sections that might apply, and ask for clarification if needed.
    - If the answer is not available in the context, respond with: "The answer is not available in the provided context."

4. **Reference Legal Sections**:
    - Where possible, refer to specific sections from the legal document or related laws.
    - Ensure the response is grounded in the provided legal context without making unsupported assumptions.

5. **Accuracy and Precision**:
    - Avoid speculating or giving incorrect information. Only answer based on the given context, and clearly state if further information is needed.

<context>
{context}
<context>

Question: {input}
"""
)

def create_or_load_faiss_index(documents, embeddings, index_path="vectorsnew.faiss"):
    if os.path.exists(index_path):
        # Load the existing FAISS index
        faiss_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        st.write("Loaded existing FAISS index.")
    else:
        # Create a new FAISS index
        faiss_index = FAISS.from_documents(documents, embeddings)
        # Save the FAISS index for future use
        faiss_index.save_local(index_path)
        st.write("Created and saved a new FAISS index.")
    return faiss_index

# Document embedding and FAISS vector store setup
def vector_embedding():
    # Load and process the text file
    text_content = load_text_file(textfile_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(text_content)
    documents = [Document(page_content=chunk) for chunk in text_chunks]

    # Load and process all PDFs in the folder
    pdf_documents = load_pdf_folder(pdffolder_path)

    for doc in pdf_documents:
        #print(doc.metadata)  # This should show the source file name or path
        pdf_chunks = text_splitter.split_text(doc.page_content)
        documents.extend([Document(page_content=chunk) for chunk in pdf_chunks])

    # Load or initialize embeddings model
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load or create FAISS index
    st.session_state.vectors = create_or_load_faiss_index(documents, st.session_state.embeddings)

prompt1 = st.text_input("Enter Your Question From Documents")

if "vectors" not in st.session_state:
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write("Response time:", time.process_time() - start)
    st.write(response['answer'])

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Display relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
