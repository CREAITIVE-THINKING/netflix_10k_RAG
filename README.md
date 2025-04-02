# Netflix 10-K RAG Explorer

An interactive application for exploring and analyzing Netflix's 10-K filings using RAG (Retrieval Augmented Generation) technology.

## Features

- Semantic search through Netflix's 10-K filings
- Dynamic query processing and intent classification
- Financial data visualization
- Section-based filtering
- Advanced context retrieval with metadata filtering
- Error handling and validation

## Technologies Used

- LangChain for RAG implementation
- OpenAI for embeddings and language model
- Pinecone for vector storage
- Streamlit for the user interface
- Python 3.10+

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/netflix_10k_RAG.git
cd netflix_10k_RAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
```

4. Run the application:
```bash
streamlit run netflix_rag_improved.py
```

## Usage

1. Enter your question about Netflix's 10-K filing in the text input
2. Use the sidebar filters to focus on specific sections or years
3. View the answer and supporting sources from the document
4. Explore financial visualizations when available

## Project Structure

- `netflix_rag_improved.py`: Main application file
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not tracked in git)
- `.gitignore`: Git ignore rules
- `README.md`: Project documentation

## Configuration

The application uses several configuration parameters that can be modified in the `Config` class:

- `MODEL_NAME`: OpenAI embedding model (default: "text-embedding-3-large")
- `LLM_MODEL`: OpenAI language model (default: "gpt-4-turbo-preview")
- `CHUNK_SIZE`: Size of document chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Security Notes

- Never commit `.env` files or API keys
- Keep your API keys secure and rotate them regularly
- Use environment variables for sensitive configuration

## License

MIT License - See LICENSE file for details 