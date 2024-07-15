# GraphRAG4OpenWebUI

<div align="center">
  <p><strong>Integrate Microsoft's GraphRAG Technology into Open WebUI for Advanced Information Retrieval</strong></p>
  English | <a href="README_ZH-CN.md">简体中文</a>
</div>

GraphRAG4OpenWebUI is an API interface specifically designed for Open WebUI, aiming to integrate Microsoft Research's GraphRAG (Graph-based Retrieval-Augmented Generation) technology. This project provides a powerful information retrieval system that supports multiple search models, particularly suitable for use in open web user interfaces.

## Project Overview

The main goal of this project is to provide a convenient interface for Open WebUI to leverage the powerful features of GraphRAG. It integrates three main retrieval methods and offers a comprehensive search option, allowing users to obtain thorough and precise search results.

### Key Retrieval Features

1. **Local Search**
   - Utilizes GraphRAG technology for efficient retrieval in local knowledge bases
   - Suitable for quick access to pre-defined structured information
   - Leverages graph structures to improve retrieval accuracy and relevance

2. **Global Search**
   - Searches for information in a broader scope, beyond local knowledge bases
   - Suitable for queries requiring more comprehensive information
   - Utilizes GraphRAG's global context understanding capabilities to provide richer search results

3. **Tavily Search**
   - Integrates external Tavily search API
   - Provides additional internet search capabilities, expanding information sources
   - Suitable for queries requiring the latest or extensive web information

4. **Full Model Search**
   - Combines all three search methods above
   - Provides the most comprehensive search results, meeting complex information needs
   - Automatically integrates and ranks information from different sources

## Installation

Ensure that you have Python 3.8 or higher installed on your system. Then, follow these steps to install:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/GraphRAG4OpenWebUI.git
   cd GraphRAG4OpenWebUI
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install fastapi uvicorn pandas tiktoken graphrag tavily-python pydantic python-dotenv asyncio aiohttp numpy scikit-learn matplotlib seaborn nltk spacy transformers torch torchvision torchaudio
   ```

   Note: The `graphrag` package might need to be installed from a specific source. If the above command fails to install `graphrag`, please refer to Microsoft Research's specific instructions or contact the maintainer for the correct installation method.

## Configuration

Before running the API, you need to set the following environment variables. You can do this by creating a `.env` file or exporting them directly in your terminal:

```bash
export GRAPHRAG_API_KEY="your_graphrag_api_key"
export TAVILY_API_KEY="your_tavily_api_key"
export GRAPHRAG_LLM_MODEL="gpt-3.5-turbo"
export GRAPHRAG_EMBEDDING_MODEL="text-embedding-3-small"
export INPUT_DIR="/path/to/your/input/directory"
```

Make sure to replace the placeholders in the above commands with your actual API keys and paths.

## Usage

1. Start the server:
   ```
   python main.py
   ```
   The server will run on `http://localhost:8012`.

2. API Endpoints:
   - `/v1/chat/completions`: POST request for performing searches
   - `/v1/models`: GET request to retrieve the list of available models

3. Integration with Open WebUI:
   In the Open WebUI configuration, set the API endpoint to `http://localhost:8012/v1/chat/completions`. This will allow Open WebUI to use the search functionality of GraphRAG4OpenWebUI.

4. Example search request:
   ```python
   import requests
   import json

   url = "http://localhost:8012/v1/chat/completions"
   headers = {"Content-Type": "application/json"}
   data = {
       "model": "full-model:latest",
       "messages": [{"role": "user", "content": "Your search query"}],
       "temperature": 0.7
   }

   response = requests.post(url, headers=headers, data=json.dumps(data))
   print(response.json())
   ```

## Available Models

- `graphrag-local-search:latest`: Local search
- `graphrag-global-search:latest`: Global search
- `tavily-search:latest`: Tavily search
- `full-model:latest`: Comprehensive search (includes all search methods above)

## Notes

- Ensure that you have the correct input files (such as Parquet files) in the `INPUT_DIR` directory.
- The API uses asynchronous programming, make sure your environment supports async operations.
- For large-scale deployment, consider using a production-grade ASGI server.
- This project is specifically designed for Open WebUI and can be easily integrated into various web-based applications.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[Apache-2.0 License](LICENSE)
