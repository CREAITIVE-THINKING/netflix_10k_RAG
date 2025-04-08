# Netflix 10-K RAG Explorer

An interactive application for exploring and analyzing Netflix's 10-K filings using RAG (Retrieval Augmented Generation) technology.

**Important Note:** This repository contains my previous custom RAG implementation, but I now recommend using Unstructured's MCP (Model Context Protocol) for document processing. You'll only need the Streamlit portion of this code combined with the Unstructured MCP workflow described below.

## Features

* Semantic search through Netflix's 10-K filings
* Dynamic query processing and intent classification
* Financial data visualization
* Section-based filtering
* Advanced context retrieval with metadata filtering
* Error handling and validation

## Technologies Used

## Technologies Used
- **Unstructured MCP** for no-code document processing and RAG pipeline setup
- Streamlit for the user interface
- Pinecone for vector storage
- OpenAI for embeddings and language model
- Python 3.10+

## Setting Up the RAG Application

### Prerequisites

- OpenAI API key
- Pinecone account and API key
- Python 3.10+
- Streamlit

### Installation

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

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
```

4. Run the application:

```bash
streamlit run netflix_rag_improved.py
```

## How to Process the Netflix 10K Document Using Unstructured MCP

This section explains how we processed the Netflix 10K filing using Unstructured's MCP (Model Context Protocol) server to create the vector database that powers this application.

### Prerequisites for Document Processing

- Claude Desktop (or another MCP client)
- Unstructured API key (get one from https://platform.unstructured.io/app/account/api-keys)
- Pinecone account for vector database storage
- Python 3.9+ installed on your system

### Step 1: Set Up AWS S3 Bucket and Credentials

Before setting up the Unstructured MCP server, you'll need to set up an AWS S3 bucket to store your documents. Unstructured has an excellent tutorial video on this process:

[Unstructured Tutorial: How to Set Up AWS Credentials and S3 Buckets](https://www.youtube.com/watch?v=hyDHfhVVAhs&t=24s)

The tutorial covers:
- Creating an AWS account if you don't have one
- Setting up an S3 bucket for document storage
- Creating IAM credentials with appropriate permissions
- Configuring your environment with AWS credentials

### Step 2: Set Up the Unstructured MCP Server

1. Clone the official Unstructured MCP server repository:
   ```bash
   git clone https://github.com/Unstructured-IO/UNS-MCP.git
   cd UNS-MCP
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   UNSTRUCTURED_API_KEY="your-unstructured-api-key"
   PINECONE_API_KEY="your-pinecone-api-key"
   AWS_ACCESS_KEY_ID="your-aws-access-key"
   AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
   AWS_REGION="your-aws-region"
   FIRECRAWL_API_KEY="your-firecrawl-api-key"  # Optional, only if using Firecrawl
   ```

4. Configure Claude Desktop to discover your MCP server:
   - Go to `~/Library/Application Support/Claude/` (create if it doesn't exist)
   - Create or edit `claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "UNS_MCP": {
         "command": "ABSOLUTE/PATH/TO/venv/bin/python",
         "args": [
           "ABSOLUTE/PATH/TO/UNS-MCP/uns_mcp/server.py"
         ],
         "disabled": false
       }
     }
   }
   ```
   - Restart Claude Desktop

### Step 3: Process the Netflix 10K Document

1. Upload the Netflix 10K PDF to a storage location (like an S3 bucket)
2. Open Claude Desktop and use the following prompts to create the workflow:

#### Create Source Connector
```
Please create an S3 source connector for the Netflix 10K document with these settings:
- Name: netflix-10k-source
- Remote URL: s3://your-bucket-name/path/to/Netflix10K.pdf
- Set recursive to true
```

After creation, you'll receive a source ID that looks like: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`

#### Create Destination Connector (make sure to setup up the index in Pinecone first)
```
Now create a Pinecone destination connector with these settings:
- Name: netflix-10k-pinecone
- Index name: netflix-10k-index
- Namespace: netflix-10k
```

After creation, you'll receive a destination ID that looks like: `yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy`

#### Create Processing Workflow
```
Create a custom workflow with the following configuration:
- Name: netflix-10k-processing
- Source ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
- Destination ID: yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy
- Workflow type: custom
- Workflow nodes:
  1. Partitioner node:
     - Type: partition
     - Subtype: vlm
     - Settings:
       - Provider: anthropic
       - Model: claude-3-5-sonnet-20241022
  2. Chunker node:
     - Type: chunk
     - Subtype: chunk_by_title
     - Settings:
       - Max characters: 1000
       - Overlap: 100
  3. Embedder node:
     - Type: embed
     - Subtype: openai
     - Settings:
       - Model name: text-embedding-3-large
```

After creation, you'll receive a workflow ID that looks like: `zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz`

#### Run the Workflow
```
Please run the workflow with ID: zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz
```

After running, you'll receive a job ID that looks like: `jjjjjjjj-jjjj-jjjj-jjjj-jjjjjjjjjjjj`

### Reference Workflow

We've created a custom workflow that:
1. Uses VLM-based partitioning with Claude 3.5 Sonnet
2. Chunks documents by title with a 1000 character limit and 200 character overlap
3. Embeds content using OpenAI's text-embedding-3-large model
4. Stores the processed data in Pinecone

This workflow effectively converts the raw Netflix 10-K PDF into a queryable vector database.

## Usage

1. Enter your question about Netflix's 10-K filing in the text input
2. Use the sidebar filters to focus on specific sections or years
3. View the answer and supporting sources from the document
4. Explore financial visualizations when available (still WIP)

## Project Structure

* `netflix_rag_improved.py`: Main application file
* `requirements.txt`: Project dependencies
* `.env`: Environment variables (not tracked in git)
* `.gitignore`: Git ignore rules
* `README.md`: Project documentation

## Configuration

The application uses several configuration parameters that can be modified in the `Config` class:
* `MODEL_NAME`: OpenAI embedding model (default: "text-embedding-3-large")
* `LLM_MODEL`: OpenAI language model (default: "gpt-4-turbo-preview")
* `CHUNK_SIZE`: Size of document chunks (default: 1000)
* `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

## Troubleshooting

- If you encounter authentication issues, verify your API keys in the `.env` file
- For workflow errors, check the job status in Unstructured Platform UI
- If embeddings are not working correctly, verify your Pinecone index dimensions match the embedding model (3072 for text-embedding-3-large)
- If the application fails to connect to Pinecone, check your environment settings and index configuration
- For AWS S3 connectivity issues:
  - Verify your AWS credentials in the `.env` file
  - Check that your IAM user has the necessary permissions (S3FullAccess or appropriate custom policy)
  - Ensure your S3 bucket has the correct permissions set
  - If using a specific region, make sure it's correctly specified in your AWS_REGION variable

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Security Notes

* Never commit `.env` files or API keys
* Keep your API keys secure and rotate them regularly
* Use environment variables for sensitive configuration

## Additional Resources

- [Unstructured Documentation](https://docs.unstructured.io/)
- [Unstructured MCP GitHub Repository](https://github.com/Unstructured-IO/UNS-MCP)
- [Model Context Protocol Documentation](https://docs.anthropic.com/claude/docs/model-context-protocol-mcp)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [AWS IAM Documentation](https://docs.aws.amazon.com/iam/)

## License

MIT License - See LICENSE file for details 
