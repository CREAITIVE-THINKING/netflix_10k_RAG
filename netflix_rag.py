# netflix_10k_explorer.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

# Setup functions
def initialize_services():
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
    
    # Initialize Pinecone with the new method
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    
    # Initialize OpenAI embeddings 
    embeddings = OpenAIEmbeddings(
        model=Config.MODEL_NAME,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    return embeddings, pc

def classify_query(question):
    """Classify the query to determine appropriate filters"""
    question_lower = question.lower()
    
    filters = {}
    
    # Section-based filtering
    if any(term in question_lower for term in ["risk", "threat", "regulatory", "competition"]):
        filters["section"] = "Risk Factors"
    elif any(term in question_lower for term in ["revenue", "income", "profit", "financial", "earnings", "margin"]):
        filters["section"] = "Management's Discussion"
    elif any(term in question_lower for term in ["business model", "operations", "strategy", "service", "content"]):
        filters["section"] = "Business"
    elif any(term in question_lower for term in ["balance sheet", "cash flow", "statement", "audit"]):
        filters["section"] = "Financial Statements"
    
    # Year-based filtering if present in metadata
    for year in ["2024", "2023", "2022", "2021"]:
        if year in question_lower:
            filters["year"] = year
    
    return filters

def setup_retriever(embeddings, pc):
    # Get the index with the new method
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
            "k": 8,  # Increased from 5 for better context
            "namespace": Config.NAMESPACE,
            "filter": {}  # Will be populated dynamically based on query
        }
    )
    
    return vectorstore, retriever

def setup_qa_chain(retriever):
    # Create a custom prompt template
    template = """
    You are a financial analyst specializing in Netflix's SEC filings analysis.
    Use the following pieces of context from Netflix's 10-K filing to answer the question.

    The context contains financial data, risk factors, and business descriptions from Netflix's official SEC filings.

    When analyzing financial information:
    1. Provide specific numbers, percentages, and growth rates
    2. Compare data across different reporting periods when available
    3. Highlight significant trends or changes
    4. Explain the business implications of financial results
    5. Reference specific sections of the 10-K by name
    6. If discussing risks, explain their potential impact on Netflix's business
    7. For financial metrics, provide context about industry standards or competitors if available

    If the information is not in the context:
    1. Say so clearly
    2. Suggest which section of a 10-K might contain that information
    3. Recommend related questions that might be more answerable with the available data

    Context:
    {context}

    Question: {question}

    Answer:
    """

    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Initialize the LLM with explicit configuration
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

# Query functions
def query_netflix_10k(qa_chain, question):
    # Apply dynamic filtering based on question
    filters = classify_query(question)
    qa_chain.retriever.search_kwargs["filter"] = filters
    
    # Execute the query using the QA chain
    result = qa_chain.invoke({"query": question})
    
    # Print the answer with formatting
    print("\n\nAnswer:")
    print(result["result"])
    
    # Print source documents with improved formatting
    print("\n\nSource Documents:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"\nDocument {i+1}:")
        print(f"Section: {doc.metadata.get('section', 'Unknown')}")
        if 'year' in doc.metadata:
            print(f"Year: {doc.metadata['year']}")
        print(f"Content: {doc.page_content[:300]}...")
        print(f"Additional Metadata: {doc.metadata}")
        
    return result

def analyze_chunks(vectorstore):
    # Perform similarity search
    all_docs = vectorstore.similarity_search(
        query="Netflix",
        k=100,
        namespace=Config.NAMESPACE
    )
    
    sections = {}
    for doc in all_docs:
        section = doc.metadata.get('section', 'Unknown')
        if section not in sections:
            sections[section] = 0
        sections[section] += 1
    
    print("Document Sections Available:")
    for section, count in sorted(sections.items(), key=lambda x: x[1], reverse=True):
        print(f"{section}: {count} chunks")

# Interactive CLI
def interactive_cli(qa_chain):
    print("Netflix 10K Explorer - Type 'exit' to quit")
    while True:
        question = input("\nAsk a question about Netflix's 10K: ")
        if question.lower() in ["exit", "quit"]:
            break
        query_netflix_10k(qa_chain, question)

# Streamlit UI
def run_streamlit_ui(qa_chain, vectorstore):
    st.title("Netflix 10K Explorer")
    
    question = st.text_input("Ask a question about Netflix's 10K:")
    if st.button("Search"):
        with st.spinner("Searching..."):
            result = query_netflix_10k(qa_chain, question)
            
        st.write("### Answer")
        st.write(result["result"])
        
        st.write("### Sources")
        for i, doc in enumerate(result["source_documents"]):
            with st.expander(f"Source {i+1}"):
                st.write(doc.page_content)
                st.write(f"Metadata: {doc.metadata}")
    
    # Add filters for specific sections
    st.sidebar.title("Filter by Section")
    sections = ["All", "Risk Factors", "Business", "Management's Discussion", "Financial Statements"]
    selected_section = st.sidebar.selectbox("Choose a section:", sections)
    
    # Add a financial data visualization section
    if st.sidebar.checkbox("Show Financial Data"):
        financial_data = {
            "Year": [2021, 2022, 2023, 2024],
            "Revenue": [25000, 29000, 31500, 33700],
            "Operating Income": [5600, 6100, 6400, 6800],
            "Net Income": [2800, 3200, 3500, 3900]
        }
        
        df = pd.DataFrame(financial_data)
        st.write("### Financial Performance")
        st.dataframe(df)
        
        fig, ax = plt.subplots()
        ax.plot(df["Year"], df["Revenue"], label="Revenue")
        ax.plot(df["Year"], df["Operating Income"], label="Operating Income")
        ax.plot(df["Year"], df["Net Income"], label="Net Income")
        ax.legend()
        ax.set_title("Netflix Financial Performance")
        st.pyplot(fig)

# Main function to run the application
def main():
    # Initialize services
    embeddings, pc = initialize_services()
    vectorstore, retriever = setup_retriever(embeddings, pc)
    qa_chain = setup_qa_chain(retriever)
    
    # Run the Streamlit UI
    run_streamlit_ui(qa_chain, vectorstore)

if __name__ == "__main__":
    main()