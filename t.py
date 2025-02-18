import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from typing import List, Set, Dict, Any
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import time
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langdetect import detect

# Load environment variables
load_dotenv()

# Configure API keys
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# Streamlit interface setup
st.title("LawAssist")

# Initialize language model
llm = ChatGroq(model="llama3-8b-8192")

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

# Enhanced prompt template
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
- Directly addresses the question
- Cites relevant legal provisions
- Maintains strict accuracy to the source materialr
- Includes all relevant details from the context 

""")

def create_or_load_faiss_index(documents, embeddings, index_path = r"code+confidencescore\vectors1.faiss" ):
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
    """Initialize vector embeddings and TRIAD framework."""
    if "vectors" not in st.session_state:
        try:
            # Load and process documents
            text_file_path = r"C:\Users\Shriya Deshpande\OneDrive\Desktop\gemma chatbot\input1.txt"
            st.session_state.docs = load_text_file(text_file_path)
            
            # Text splitting with optimized parameters
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(st.session_state.docs)
            documents = [Document(page_content=chunk) for chunk in chunks]
            
            # Initialize embeddings and vector store
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            # Load or create FAISS index
            st.session_state.vectors = create_or_load_faiss_index(documents, st.session_state.embeddings)

            # Initialize TRIAD framework
            st.session_state.triad = TriadFramework(st.session_state.embeddings)
            
            return True
        except Exception as e:
            st.error(f"Error in vector embedding: {e}")
            return False

# Add translation functions
def detect_language(text: str) -> str:
    """Detect the language of input text."""
    try:
        return detect(text)
    except:
        return 'en'  # Default to English if detection fails

def translate_to_english(text: str, source_lang: str) -> str:
    """Translate text to English."""
    if source_lang == 'en':
        return text
    translator = GoogleTranslator(source=source_lang, target='en')
    return translator.translate(text)

def translate_answer(answer: str, target_language: str) -> str:
    """Translate answer to target language."""
    if target_language == 'en':
        return answer
    translator = GoogleTranslator(source='en', target=target_language)
    return translator.translate(answer)

# Modified process_query function with translation support
def process_query(query: str, retrieval_chain: Any, triad_framework: TriadFramework) -> Dict:
    """Process query and generate response with metrics, including translation."""
    try:
        # Detect original language
        original_language = detect_language(query)
        
        # Translate query to English if needed
        if original_language != 'en':
            english_query = translate_to_english(query, original_language)
        else:
            english_query = query

        # Get initial response using English query
        response = retrieval_chain.invoke({'input': english_query})
        
        if not response.get('context'):
            return {
                'answer': "No relevant context found to answer the question.",
                'metrics': {'precision': 0, 'recall': 0, 'f1_score': 0},
                'context': []
            }
        
        # Filter and process context
        filtered_context = triad_framework.filter_relevant_context(english_query, response["context"])
        combined_context = " ".join([doc.page_content for doc in filtered_context])
        
        # Calculate metrics
        metrics = triad_framework.calculate_metrics(english_query, response['answer'], combined_context)
        
        # Translate answer back to original language if needed
        if original_language != 'en':
            translated_answer = translate_answer(response['answer'], original_language)
        else:
            translated_answer = response['answer']
        
        return {
            'answer': translated_answer,
            'metrics': metrics,
            'context': filtered_context,
            'original_language': original_language
        }
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return {
            'answer': "An error occurred while processing your query.",
            'metrics': {'precision': 0, 'recall': 0, 'f1_score': 0},
            'context': [],
            'original_language': 'en'
        }

# Modified display_results function to show language information
def display_results(results: Dict):
    """Display results with metrics, relevant context, and language information."""
    st.write("### Answer:")
    if results.get('original_language') != 'en':
        st.write(f"[Response in {results['original_language']}]")
    st.write(results['answer'])
    
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

# The rest of your code remains the same until the main interface

# Main interface
st.write("You can ask questions in any language - the system will automatically detect and translate.")
prompt1 = st.text_input("Enter Your Question From Documents:")

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
            # Create processing chain
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Process query and measure time
            start_time = time.time()
            results = process_query(prompt1, retrieval_chain, st.session_state.triad)
            process_time = time.time() - start_time
            
            # Display results
            display_results(results)
            st.write(f"Processing Time: {process_time:.2f} seconds")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
