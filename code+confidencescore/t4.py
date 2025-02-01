#confidence score
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
from typing import List
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Load API keys
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("LawAssist")

llm = ChatGroq(model="gemma2-9b-it")

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# TRIAD Framework Implementation
class TriadFramework:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.relevance_threshold = 0.7
        
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using embeddings."""
        try:
            emb1 = self.embeddings.embed_query(text1)
            emb2 = self.embeddings.embed_query(text2)
            return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def filter_relevant_context(self, query: str, contexts: List[Document]) -> List[Document]:
        """Filter contexts based on relevance to the query."""
        relevant_contexts = []
        for context in contexts:
            similarity = self.calculate_semantic_similarity(query, context.page_content)
            if similarity >= self.relevance_threshold:
                relevant_contexts.append(context)
        return relevant_contexts
    
    def verify_answer_relevance(self, query: str, answer: str, context: str) -> tuple:
        """Verify if the answer is relevant to both query and context."""
        query_similarity = self.calculate_semantic_similarity(query, answer)
        context_similarity = self.calculate_semantic_similarity(context, answer)
        
        is_relevant = query_similarity >= self.relevance_threshold and context_similarity >= self.relevance_threshold
        confidence_score = (query_similarity + context_similarity) / 2
        
        return is_relevant, confidence_score

# Enhanced prompt template with TRIAD considerations
prompt = ChatPromptTemplate.from_template("""
                                          
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
                                          
Please provide a response that:
- Directly answers the question
- Cites specific parts of the context
- Maintains high relevance to both question and context

""")

def create_or_load_faiss_index(documents, embeddings, index_path="vectors1.faiss"):
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


def vector_embedding():
    if "vectors" not in st.session_state:
        text_file_path = r"C:\Users\Shriya Deshpande\OneDrive\Desktop\gemma chatbot\input1.txt"
        
        st.session_state.docs = load_text_file(text_file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(st.session_state.docs)
        documents = [Document(page_content=chunk) for chunk in chunks]
        
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
       # Load or create FAISS index
        st.session_state.vectors = create_or_load_faiss_index(documents, st.session_state.embeddings)
       # Initialize TRIAD framework
        st.session_state.triad = TriadFramework(st.session_state.embeddings)

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
    
    # Get initial response
    response = retrieval_chain.invoke({'input': prompt1})
    
    # Apply TRIAD framework
    if 'triad' in st.session_state:
        # Filter relevant context
        filtered_context = st.session_state.triad.filter_relevant_context(
            prompt1, 
            response["context"]
        )
        
        # Verify answer relevance
        is_relevant, confidence = st.session_state.triad.verify_answer_relevance(
            prompt1,
            response['answer'],
            " ".join([doc.page_content for doc in filtered_context])
        )
        
        process_time = time.process_time() - start
        
        # Display results
        st.write("### Answer:")
        st.write(response['answer'])
        
        st.write("### Confidence Score:")
        st.progress(confidence)
        st.write(f"Confidence: {confidence:.2f}")
        
        st.write("### Processing Time:")
        st.write(f"{process_time:.2f} seconds")
        
        # Display relevant context
        with st.expander("Relevant Context"):
            for i, doc in enumerate(filtered_context):
                st.write(f"Chunk {i+1}:")
                st.write(doc.page_content)
                st.write("--------------------------------")

