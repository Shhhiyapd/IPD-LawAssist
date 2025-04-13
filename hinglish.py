import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from typing import List, Set, Dict, Any
import numpy as np
import time
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables
load_dotenv()

# Configure API keys
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Streamlit interface setup
st.title("LawAssist")

# Initialize language model with LangChain wrapper
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

class TriadFramework:
    """
    TRIAD Framework for enhanced document analysis and response generation
    with precision and recall metrics
    """
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.relevance_threshold = 0.7
        self.cache = {}
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using embeddings."""
        try:
            # Create cache key
            cache_key = f"{hash(text1)}-{hash(text2)}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Calculate embeddings and similarity
            emb1 = self.embeddings.embed_query(text1)
            emb2 = self.embeddings.embed_query(text2)
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            # Cache result
            self.cache[cache_key] = similarity
            return similarity
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def get_key_terms(self, text: str) -> Set[str]:
        """Extract key terms from text for precision/recall calculation."""
        # Common legal terms to preserve
        legal_terms = {'section', 'act', 'law', 'court', 'rights', 'penalty', 'punishment'}
        
        # Enhanced stop words list
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after'
        }
        
        # Convert to lowercase and split
        terms = set(text.lower().split())
        
        # Remove stop words but preserve legal terms
        filtered_terms = {term for term in terms if term in legal_terms or term not in stop_words}
        return filtered_terms
    
    def calculate_metrics(self, query: str, answer: str, context: str) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score with enhanced accuracy."""
        # Get key terms
        query_terms = self.get_key_terms(query)
        answer_terms = self.get_key_terms(answer)
        context_terms = self.get_key_terms(context)
        
        # Combined relevant terms
        relevant_terms = query_terms.union(context_terms)
        
        # Calculate metrics
        if len(answer_terms) == 0:
            precision = 0
        else:
            precision = len(answer_terms.intersection(relevant_terms)) / len(answer_terms)
        
        if len(relevant_terms) == 0:
            recall = 0
        else:
            recall = len(answer_terms.intersection(relevant_terms)) / len(relevant_terms)
        
        # Calculate F1 score
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def filter_relevant_context(self, query: str, contexts: List[Document]) -> List[Document]:
        """Filter and rank contexts based on relevance to the query."""
        scored_contexts = []
        for context in contexts:
            similarity = self.calculate_semantic_similarity(query, context.page_content)
            if similarity >= self.relevance_threshold:
                scored_contexts.append((context, similarity))
        
        # Sort by similarity score
        scored_contexts.sort(key=lambda x: x[1], reverse=True)
        return [context for context, _ in scored_contexts]

def load_text_file(file_path: str) -> str:
    """Load and read text file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return ""


def create_or_load_faiss_index(documents, embeddings, index_path = r"C:\ProjectIPD\IPD-LawAssist\labse_vectors.faiss"):
    # Try to load existing index
    if os.path.exists(index_path):
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
    """Initialize vector embeddings and TRIAD framework."""
    if "vectors" not in st.session_state:
        try:
            # Load and process documents
            text_file_path = r"C:\ProjectIPD\IPD-LawAssist\input1.txt"
            st.session_state.docs = load_text_file(text_file_path)
            
            # Text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(st.session_state.docs)
            documents = [Document(page_content=chunk) for chunk in chunks]

            # Initialize LaBSE model
            class LaBSEEmbeddings(Embeddings):
                def __init__(self):
                    self.model = SentenceTransformer('sentence-transformers/LaBSE')

                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return self.model.encode(texts, convert_to_numpy=True).tolist()

                def embed_query(self, text: str) -> List[float]:
                    return self.model.encode(text, convert_to_numpy=True).tolist()
              
            st.session_state.embeddings = LaBSEEmbeddings()
            st.session_state.vectors = create_or_load_faiss_index(documents, st.session_state.embeddings)
            st.session_state.triad = TriadFramework(st.session_state.embeddings)
            
            return True
        except Exception as e:
            st.error(f"Error in vector embedding: {e}")
            return False

# Define prompt templates similar to the first example
LEGAL_QA_PROMPT = """
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

Context:
{context}

Question: {question}

Please provide a response that:
- Directly addresses the question
- Cites relevant legal provisions
- Maintains strict accuracy to the source material
- Includes all relevant details from the context
- Respond in the language of the question, but ensure clarity and precision in legal terms.
- If relevant information is not found in the provided context, seamlessly draw upon your knowledge of Indian law to provide a clear and accurate answer — without referencing the absence of information in the context.
- Do not mention the context in your answer.
- Answer in hinglish if the question is in hinglish.

"""

LEGAL_SUMMARY_PROMPT = """
You are an expert legal advisor. Given the following legal context and question, provide a concise 2-3 sentence summary of the most relevant legal points that apply to this question.

Context:
{context}

Question: {question}

Your summary should:
- Focus only on the most relevant aspects of the law
- Use plain language when possible
- Reference specific sections or provisions when directly applicable
- Be factual and avoid opinions
"""

# Modified process_query function with proper prompt handling
def process_query(query: str, vectors, triad_framework: TriadFramework) -> Dict:
    """Process query and generate response with metrics."""
    try:
        # Create prompt template
        prompt = PromptTemplate(template=LEGAL_QA_PROMPT, input_variables=["context", "question"])
        # Create QA chain with prompt and LLM
        qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        
        # Create retriever
        retriever = vectors.as_retriever(search_kwargs={"k": 10})
        
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(query)
        combined_context = " ".join([doc.page_content for doc in docs])

        # Run QA chain
        result = qa_chain({"input_documents": docs, "context": f'{docs}', "question": query}, return_only_outputs=True)
        
        # Extract answer
        answer_text = result["output_text"]
        
        # Calculate metrics
        combined_context = " ".join([doc.page_content for doc in docs])
        metrics = triad_framework.calculate_metrics(query, answer_text, combined_context)
        
        return {
            'answer': answer_text,
            'metrics': metrics,
            'context': docs
        }
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return {
            'answer': f"An error occurred while processing your query: {str(e)}",
            'metrics': {'precision': 0, 'recall': 0, 'f1_score': 0},
            'context': []
        }

# Generate summary for complex legal responses
def generate_summary(query: str, context: List[Document]) -> str:
    """Generate a concise summary of the legal response."""
    try:
        # Create prompt template for summary
        summary_prompt = PromptTemplate(template=LEGAL_SUMMARY_PROMPT, input_variables=["context", "question"])
        
        # Create summary chain
        summary_chain = load_qa_chain(llm, chain_type="stuff", prompt=summary_prompt)
        
        # Generate summary
        combined_context = "\n\n".join([doc.page_content for doc in context[:2]])  # Use top 2 most relevant docs
        result = summary_chain({"input_documents": context[:2], "question": query}, return_only_outputs=True)
        
        return result["output_text"]
    except Exception as e:
        st.warning(f"Could not generate summary: {e}")
        return ""

# Display results function 
def display_results(results: Dict, query: str):
    """Display results with metrics, summary and relevant context."""
    # Main answer
    st.markdown("### Answer:")
    formatted_answer = results['answer'].replace('\n', '\n\n')
    st.markdown(formatted_answer)
    
    # Generate and display summary if answer is long
    if len(results['answer']) > 500 and results['context']:
        summary = generate_summary(query, results['context'])
        if summary:
            st.markdown("### Summary:")
            st.markdown(summary)
    
    # Display metrics
    st.write("### Evaluation Metrics:")
    metrics = results['metrics']
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    # Display metrics with color-coded bars
    with col1:
        st.write("Precision:")
        st.progress(metrics['precision'])
        st.write(f"{metrics['precision']:.2%}")
        
    with col2:
        st.write("Recall:")
        st.progress(metrics['recall'])
        st.write(f"{metrics['recall']:.2%}")
        
    with col3:
        st.write("F1 Score:")
        st.progress(metrics['f1_score'])
        st.write(f"{metrics['f1_score']:.2%}")
    
    # Display relevant context
    with st.expander("Relevant Legal Context"):
        for i, doc in enumerate(results['context']):
            st.write(f"Reference {i+1}:")
            st.write(doc.page_content)
            st.write("---")

# Main interface
st.write("You can ask questions in any language - the system will automatically detect and translate.")
prompt1 = st.text_input("Enter Your Question About Legal Documents:")

if "vectors" not in st.session_state:
    with st.spinner("Initializing document analysis system..."):
        if vector_embedding():
            st.success("System initialized successfully!")
        else:
            st.error("Failed to initialize system. Please check the error message above.")

# Process query when submitted
if prompt1:
    with st.spinner("Processing your query..."):
        try:
            # Process query and measure time
            start_time = time.time()
            results = process_query(prompt1, st.session_state.vectors, st.session_state.triad)
            process_time = time.time() - start_time
            
            # Display results
            display_results(results, prompt1)
            st.write(f"Processing Time: {process_time:.2f} seconds")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")