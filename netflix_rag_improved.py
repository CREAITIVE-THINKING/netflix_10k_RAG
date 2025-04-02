"""
Improved Netflix 10-K RAG Implementation
Using Unstructured workflow approach for better document processing and retrieval
"""
import os
import re
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema.retriever import BaseRetriever
from langchain.schema import Document
from pydantic import Field
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Load environment variables
    load_dotenv()
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    INDEX_NAME = "netflix-10k-index"
    NAMESPACE = "netflix-10k"
    MODEL_NAME = "text-embedding-3-large"
    LLM_MODEL = "gpt-4-turbo-preview"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
# Validation functions
def validate_config():
    """Validate that all required configuration variables are set"""
    required_vars = {
        "OPENAI_API_KEY": Config.OPENAI_API_KEY,
        "PINECONE_API_KEY": Config.PINECONE_API_KEY,
        "INDEX_NAME": Config.INDEX_NAME
    }
    
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")

def check_pinecone_connection(pc):
    """Check if Pinecone connection is healthy and index exists"""
    try:
        index = pc.Index(Config.INDEX_NAME)
        stats = index.describe_index_stats()
        logger.info(f"Connected to Pinecone index with {stats['total_vector_count']} vectors")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone: {e}")
        return False

def verify_embedding_dimensions(embeddings, pc):
    """Verify that embedding dimensions match index expectations"""
    try:
        test_text = "Dimension verification test"
        test_embedding = embeddings.embed_query(test_text)
        index = pc.Index(Config.INDEX_NAME)
        index_stats = index.describe_index_stats()
        
        # Some Pinecone indexes don't explicitly report dimension
        dimension = index_stats.get('dimension')
        if dimension and len(test_embedding) != dimension:
            raise ValueError(
                f"Embedding dimension mismatch: Model produces {len(test_embedding)}-d "
                f"vectors but index expects {dimension}-d vectors"
            )
        logger.info(f"Verified embedding dimensions: {len(test_embedding)}")
    except Exception as e:
        logger.error(f"Embedding dimension verification failed: {e}")
        raise

# Preprocessing utilities
class TextPreprocessor:
    @staticmethod
    def clean_text(text):
        """Remove excessive whitespace and normalize formatting"""
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def extract_sections(text):
        """Extract major sections from the 10-K document"""
        # Identify major section headers (customize based on the document structure)
        section_pattern = r'(?:\n|^)((?:[A-Z][A-Z\s]+)|(?:Item\s+\d+[A-Za-z\s\.\-]+))(?:\n)'
        sections = re.split(section_pattern, text)
        
        # Pair section titles with content
        structured_sections = []
        for i in range(1, len(sections), 2):
            if i < len(sections) - 1:
                section_title = sections[i].strip()
                section_content = sections[i+1].strip()
                
                structured_sections.append({
                    "title": section_title,
                    "content": section_content
                })
        
        return structured_sections
    
    @staticmethod
    def identify_tables(text):
        """Identify and mark table structures in the text"""
        # Simple pattern to identify tables (customize based on actual table format)
        table_pattern = r'(?:\n|^)(\s*[-]+\s*[|]\s*[-]+\s*)+(?:\n)'
        
        # Mark tables for special processing
        marked_text = re.sub(table_pattern, "\n<TABLE>\n", text)
        return marked_text

    @staticmethod
    def normalize_region_names(text):
        """Normalize region names for consistent retrieval"""
        # Map of region name variations
        region_maps = {
            r'\bU\.S\.\b': 'United States',
            r'\bUS\b': 'United States',
            r'\bUnited States\b': 'UCAN',
            r'\bUCAN\b': 'United States and Canada',
            r'\bAPAC\b': 'Asia-Pacific',
            r'\bEMEA\b': 'Europe, Middle East, and Africa',
            r'\bLATAM\b': 'Latin America'
        }
        
        # Apply replacements
        normalized_text = text
        for pattern, replacement in region_maps.items():
            normalized_text = re.sub(pattern, replacement, normalized_text)
            
        return normalized_text

# Semantic chunker for better context preservation
class SemanticChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_by_section(self, sections):
        """Create chunks based on document sections with appropriate metadata"""
        all_chunks = []
        
        for section in sections:
            section_title = section["title"]
            content = section["content"]
            
            # Further split long sections
            if len(content) > self.chunk_size:
                chunks = self._split_text(content, section_title)
                all_chunks.extend(chunks)
            else:
                # Keep short sections as single chunks
                all_chunks.append({
                    "text": content,
                    "metadata": {
                        "section": section_title,
                        "subsection": "main",
                        "chunk_type": "section"
                    }
                })
        
        return all_chunks
    
    def _split_text(self, text, section_title):
        """Split long text into semantically meaningful chunks"""
        chunks = []
        
        # Try to split on paragraph boundaries
        paragraphs = text.split('\n\n')
        current_chunk = ""
        current_subsection = "intro"
        
        for i, para in enumerate(paragraphs):
            # Check for subsection headers
            if re.match(r'^[A-Z][A-Za-z\s]+$', para.strip()) and len(para) < 100:
                current_subsection = para.strip()
            
            # If adding the paragraph exceeds chunk size and we have content,
            # create a chunk and start a new one
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk,
                    "metadata": {
                        "section": section_title,
                        "subsection": current_subsection,
                        "chunk_type": "paragraph_group"
                    }
                })
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_text = " ".join(words[-self.chunk_overlap:]) if len(words) > self.chunk_overlap else ""
                current_chunk = overlap_text + "\n\n" + para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "metadata": {
                    "section": section_title,
                    "subsection": current_subsection,
                    "chunk_type": "paragraph_group"
                }
            })
        
        return chunks
    
    def process_financial_tables(self, text, section_title):
        """Special processing for financial tables to keep them intact"""
        # Find table markers
        table_chunks = []
        
        # Split by table markers
        parts = text.split("<TABLE>")
        
        for i, part in enumerate(parts):
            if i == 0 and part:  # Text before first table
                table_chunks.append({
                    "text": part,
                    "metadata": {
                        "section": section_title,
                        "chunk_type": "text"
                    }
                })
            elif part:  # Table content
                table_chunks.append({
                    "text": part,
                    "metadata": {
                        "section": section_title,
                        "chunk_type": "table",
                        "table_index": i
                    }
                })
        
        return table_chunks

# Query processing for better retrieval
class QueryProcessor:
    @staticmethod
    def expand_query(query):
        """Expand query with synonyms and related terms for better retrieval"""
        # Lowercase for consistent matching
        query_lower = query.lower()
        
        # Maps for query expansion
        term_maps = {
            "revenue": ["revenue", "earnings", "income", "sales"],
            "profit": ["profit", "income", "earnings", "margin"],
            "subscriber": ["subscriber", "member", "membership", "user"],
            "us": ["us", "u.s.", "united states", "ucan", "united states and canada"],
            "growth": ["growth", "increase", "rise", "change"]
        }
        
        # Create expansion terms
        expanded_terms = []
        for term, expansions in term_maps.items():
            if term in query_lower:
                for exp in expansions:
                    if exp not in expanded_terms and exp != term:
                        expanded_terms.append(exp)
        
        # Create expanded query
        expanded_query = query
        if expanded_terms:
            expanded_query += " " + " ".join(expanded_terms)
            
        return expanded_query
    
    @staticmethod
    def classify_query_intent(query):
        """Classify query to determine retrieval strategy"""
        query_lower = query.lower()
        
        # Simple intent classification
        intents = {
            "financial_data": any(term in query_lower for term in ["revenue", "profit", "income", "earnings", "financial", "money"]),
            "regional_data": any(term in query_lower for term in ["region", "country", "us", "u.s.", "united states", "international"]),
            "growth_metrics": any(term in query_lower for term in ["growth", "increase", "change", "compare", "trend"]),
            "subscriber_info": any(term in query_lower for term in ["subscriber", "member", "user", "customer", "audience"]),
        }
        
        # Determine primary and secondary intents
        primary_intent = max(intents.items(), key=lambda x: x[1])[0] if any(intents.values()) else "general"
        
        return {
            "primary_intent": primary_intent,
            "all_intents": [k for k, v in intents.items() if v],
            "is_financial": intents.get("financial_data", False),
            "is_regional": intents.get("regional_data", False)
        }
    
    @staticmethod
    def create_metadata_filter(query_intent):
        """Create metadata filter based on query intent"""
        filters = {}
        
        # Add relevant filters based on intent
        if query_intent["is_financial"]:
            filters["chunk_type"] = {"$in": ["table", "paragraph_group"]}
            
        if "financial_data" in query_intent["all_intents"]:
            filters["section"] = {"$in": ["Management's Discussion", "Financial Statements", "Financial Highlights"]}
            
        if "subscriber_info" in query_intent["all_intents"]:
            filters["section"] = {"$in": ["Business", "Management's Discussion"]}
        
        return filters if filters else {}

# Setup functions
def initialize_services():
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
    
    # Initialize Pinecone
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    
    # Initialize OpenAI embeddings (without proxy parameter)
    embeddings = OpenAIEmbeddings(
        model=Config.MODEL_NAME,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    return embeddings, pc

def setup_retriever(embeddings, pc, query_processor):
    # Get the index
    index = pc.Index(Config.INDEX_NAME)
    
    # Initialize the vector store
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )
    
    # Create a retriever with improved search parameters
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 8,  # Increased from 5 for better coverage
            "namespace": Config.NAMESPACE,
            "filter": {}  # Will be dynamically updated
        }
    )
    
    # Create a dynamic retriever that processes queries before retrieval
    class DynamicRetriever(BaseRetriever):
        """Dynamic retriever that processes queries before retrieval"""
        
        base_retriever: BaseRetriever = Field(description="Base retriever to wrap")
        query_processor: Any = Field(description="Query processor for expanding and classifying queries")
        search_kwargs: Dict = Field(default_factory=dict, description="Search parameters")
        
        class Config:
            """Pydantic config"""
            arbitrary_types_allowed = True
        
        def get_relevant_documents(self, query: str) -> List[Document]:
            """Process the query and get relevant documents"""
            try:
                # Process and expand the query
                expanded_query = self.query_processor.expand_query(query)
                
                # Classify query intent
                query_intent = self.query_processor.classify_query_intent(query)
                
                # Create metadata filter based on intent
                metadata_filter = self.query_processor.create_metadata_filter(query_intent)
                
                # Update retriever filter
                self.base_retriever.search_kwargs["filter"] = metadata_filter
                
                # Get results with expanded query
                results = self.base_retriever.get_relevant_documents(expanded_query)
                
                # If no results with filter, try without filter
                if not results:
                    self.base_retriever.search_kwargs["filter"] = {}
                    results = self.base_retriever.get_relevant_documents(expanded_query)
                
                return results
            except Exception as e:
                logger.error(f"Error in get_relevant_documents: {e}")
                return []
    
    # Create dynamic retriever
    dynamic_retriever = DynamicRetriever(
        base_retriever=retriever,
        query_processor=query_processor,
        search_kwargs=retriever.search_kwargs
    )
    
    return vectorstore, dynamic_retriever

def setup_qa_chain(retriever):
    # Create an improved prompt template
    template = """
    You are a financial analyst specializing in Netflix's SEC filings and financial data.
    
    Use the following pieces of context from Netflix's 10-K filing to answer the question.
    
    Important notes about the data:
    - All financial figures are in thousands of dollars unless specified otherwise
    - "UCAN" stands for "United States and Canada" region
    - When questions mention "U.S." or "United States", use the UCAN (United States and Canada) data
    - Financial tables contain the most accurate numerical data
    - Average revenue per paying membership is calculated by dividing revenue by the average number of paying memberships
    
    When analyzing financial information:
    - Always provide specific numbers with proper context
    - Compare current data with previous periods when available
    - Identify trends and growth rates
    - Explain what the numbers mean for Netflix's business
    
    If you don't find the exact information in the context, explain what information is available and what would be needed for a complete answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """

    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Initialize the LLM with lower temperature for more factual responses
    llm = ChatOpenAI(
        model_name=Config.LLM_MODEL,
        temperature=0,
        openai_api_key=Config.OPENAI_API_KEY
    )

    # Set up conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="query",
        output_key="result"
    )

    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PROMPT,
            "verbose": True
        }
    )
    
    return qa_chain

# Document indexing function
def index_document(file_path, embeddings, pc):
    """Process and index a new document"""
    # Read document
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # Initialize processors
    preprocessor = TextPreprocessor()
    chunker = SemanticChunker(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    # Process document
    cleaned_text = preprocessor.clean_text(raw_text)
    marked_text = preprocessor.identify_tables(cleaned_text)
    normalized_text = preprocessor.normalize_region_names(marked_text)
    sections = preprocessor.extract_sections(normalized_text)
    
    # Create chunks
    chunks = chunker.chunk_by_section(sections)
    
    # Connect to Pinecone
    index = pc.Index(Config.INDEX_NAME)
    
    # Process and upsert chunks
    upsert_batch = []
    for i, chunk in enumerate(chunks):
        # Create embedding
        embedding = embeddings.embed_query(chunk["text"])
        
        # Create record
        record = {
            "id": f"netflix-10k-{i}",
            "values": embedding,
            "metadata": chunk["metadata"],
            "text": chunk["text"]
        }
        
        upsert_batch.append(record)
        
        # Batch upsert every 100 records
        if len(upsert_batch) >= 100:
            index.upsert(vectors=upsert_batch, namespace=Config.NAMESPACE)
            upsert_batch = []
    
    # Upsert any remaining records
    if upsert_batch:
        index.upsert(vectors=upsert_batch, namespace=Config.NAMESPACE)
    
    print(f"Indexed {len(chunks)} chunks from the document")

# Query function
def query_netflix_10k(qa_chain, question):
    """Query the Netflix 10-K data with improved error handling"""
    try:
        # Execute the query using the QA chain with invoke method
        result = qa_chain.invoke({"query": question})
        
        # Return the result and source documents
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }
    except Exception as e:
        logger.error(f"Error querying Netflix 10-K: {e}")
        if "dimension" in str(e):
            return {
                "answer": "Error: Embedding dimension mismatch. Please check model configuration.",
                "sources": []
            }
        elif "PineconeException" in str(e.__class__.__name__):
            return {
                "answer": "Error: Unable to connect to knowledge base. Please try again later.",
                "sources": []
            }
        else:
            return {
                "answer": f"An error occurred: {str(e)}",
                "sources": []
            }

# Streamlit UI
def run_streamlit_ui():
    st.title("Netflix 10K Explorer")
    st.write("Ask questions about Netflix's financial data from their 10-K filings")
    
    try:
        # Initialize services
        embeddings, pc = initialize_services()
        query_processor = QueryProcessor()
        vectorstore, retriever = setup_retriever(embeddings, pc, query_processor)
        qa_chain = setup_qa_chain(retriever)
        
        # User query input
        question = st.text_input("Ask a question about Netflix's 10-K:")
        
        if st.button("Search"):
            if question:
                with st.spinner("Analyzing Netflix's financial data..."):
                    try:
                        # Query the QA chain
                        result = query_netflix_10k(qa_chain, question)
                        
                        if "error" in result.get("answer", "").lower():
                            st.error(result["answer"])
                        else:
                            # Display answer
                            st.subheader("Answer")
                            st.write(result["answer"])
                            
                            # Display sources
                            if result["sources"]:
                                st.subheader("Sources")
                                for i, doc in enumerate(result["sources"]):
                                    with st.expander(f"Source {i+1} - {doc.metadata.get('section', 'Unknown Section')}"):
                                        st.write("**Content:**")
                                        st.write(doc.page_content)
                                        st.write("**Metadata:**")
                                        st.json(doc.metadata)
                    except Exception as e:
                        logger.error(f"Error processing query: {e}")
                        st.error(f"Error processing your query: {str(e)}")
        
        # Sidebar filters
        st.sidebar.title("Filters & Options")
        
        # Section filter
        st.sidebar.subheader("Filter by Section")
        sections = ["All", "Financial Statements", "Risk Factors", "Management's Discussion", "Business"]
        selected_section = st.sidebar.selectbox("Choose a section:", sections)
        
        # Year filter
        st.sidebar.subheader("Filter by Year")
        years = ["All", "2024", "2023", "2022"]
        selected_year = st.sidebar.selectbox("Choose a year:", years)
        
        # Advanced options
        st.sidebar.subheader("Advanced Options")
        context_size = st.sidebar.slider("Context Size", min_value=3, max_value=15, value=8, 
                                       help="Number of document chunks to include in the context")
        
        # About section
        st.sidebar.title("About")
        st.sidebar.info("""
        This application uses RAG (Retrieval-Augmented Generation) to answer questions about Netflix's 10-K filings.
        It leverages semantic document processing, dynamic query expansion, and financial-specific optimizations to provide accurate answers.
        """)
    except Exception as e:
        logger.error(f"Error in Streamlit UI: {e}")
        st.error(f"Application error: {str(e)}")
        st.info("Please check your configuration and try restarting the application.")

# Main function to run the application
def main():
    try:
        # Validate configuration
        validate_config()
        
        # Initialize services
        embeddings, pc = initialize_services()
        
        # Check Pinecone connection
        if not check_pinecone_connection(pc):
            st.error("Failed to connect to Pinecone. Please check your configuration.")
            return
        
        # Verify embedding dimensions
        verify_embedding_dimensions(embeddings, pc)
        
        # Run the Streamlit UI
        run_streamlit_ui()
    except Exception as e:
        # Log the error
        logger.error(f"Application startup error: {e}")
        
        # Display error in UI if using Streamlit
        if 'st' in globals():
            st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()