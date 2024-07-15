# GraphRAG4OpenWebUI

GraphRAG4OpenWebUI 是一个专为 Open WebUI 设计的 API 接口，旨在集成微软研究院的 GraphRAG（Graph-based Retrieval-Augmented Generation）技术。该项目提供了一个强大的信息检索系统，支持多种搜索模型，特别适合在开放式 Web 用户界面中使用。

## 项目概述

本项目的主要目标是为 Open WebUI 提供一个便捷的接口，以利用 GraphRAG 的强大功能。它集成了三种主要的检索方法，并提供了一个综合搜索选项，使用户能够获得全面而精确的搜索结果。

### 主要检索功能

1. **本地搜索（Local Search）**
   - 利用 GraphRAG 技术在本地知识库中进行高效检索
   - 适用于快速访问预先定义的结构化信息
   - 利用图结构提高检索的准确性和相关性

2. **全局搜索（Global Search）**
   - 在更广泛的范围内搜索信息，超越本地知识库的限制
   - 适用于需要更全面信息的查询
   - 利用 GraphRAG 的全局上下文理解能力，提供更丰富的搜索结果

3. **Tavily 搜索**
   - 集成外部 Tavily 搜索 API
   - 提供额外的互联网搜索能力，扩展信息源
   - 适用于需要最新或广泛网络信息的查询

4. **全模型搜索（Full Model Search）**
   - 综合上述三种搜索方法
   - 提供最全面的搜索结果，满足复杂的信息需求
   - 自动整合和排序来自不同来源的信息

## 安装

确保你的系统中已安装 Python 3.8 或更高版本。然后，按照以下步骤安装：

1. 克隆仓库：

   ```
   git clone https://github.com/your-username/GraphRAG4OpenWebUI.git
   cd GraphRAG4OpenWebUI
   ```

2. 创建并激活虚拟环境：

   ```
   python -m venv venv
   source venv/bin/activate  # 在 Windows 上使用 venv\Scripts\activate
   ```

3. 安装依赖：

   ```
   pip install fastapi uvicorn pandas tiktoken graphrag tavily-python pydantic python-dotenv asyncio aiohttp numpy scikit-learn matplotlib seaborn nltk spacy transformers torch torchvision torchaudio
   ```

   注意：`graphrag` 包可能需要从特定的源安装。如果上述命令无法安装 `graphrag`，请参考微软研究院的具体说明或联系维护者获取正确的安装方法。

## 配置

在运行 API 之前，需要设置以下环境变量。你可以通过创建 `.env` 文件或直接在终端中导出这些变量：

```bash
export GRAPHRAG_API_KEY="your_graphrag_api_key"
export TAVILY_API_KEY="your_tavily_api_key"
export GRAPHRAG_LLM_MODEL="gpt-3.5-turbo"
export GRAPHRAG_EMBEDDING_MODEL="text-embedding-3-small"
export INPUT_DIR="/path/to/your/input/directory"
```

确保将上述命令中的占位符替换为实际的 API 密钥和路径。

## 使用方法

1. 启动服务器：

   ```
   python main.py
   ```

   服务器将在 `http://localhost:8012` 上运行。

2. API 端点：

   - `/v1/chat/completions`: POST 请求，用于执行搜索
   - `/v1/models`: GET 请求，获取可用模型列表

3. 在 Open WebUI 中集成：

   在 Open WebUI 的配置中，将 API 端点设置为 `http://localhost:8012/v1/chat/completions`。这将允许 Open WebUI 使用 GraphRAG4OpenWebUI 的搜索功能。

4. 发送搜索请求示例：

   ```python
   import requests
   import json

   url = "http://localhost:8012/v1/chat/completions"
   headers = {"Content-Type": "application/json"}
   data = {
       "model": "full-model:latest",
       "messages": [{"role": "user", "content": "你的搜索查询"}],
       "temperature": 0.7
   }

   response = requests.post(url, headers=headers, data=json.dumps(data))
   print(response.json())
   ```

## 可用模型

- `graphrag-local-search:latest`: 本地搜索
- `graphrag-global-search:latest`: 全局搜索
- `tavily-search:latest`: Tavily 搜索
- `full-model:latest`: 综合搜索（包含上述所有搜索方法）

## 注意事项

- 确保在 `INPUT_DIR` 目录中有正确的输入文件（如 Parquet 文件）。
- API 使用异步编程，确保你的环境支持异步操作。
- 对于大规模部署，考虑使用生产级的 ASGI 服务器。
- 本项目专为 Open WebUI 设计，可以轻松集成到各种基于 Web 的应用中。

## 贡献

欢迎提交 Pull Requests 来改进这个项目。对于重大变更，请先开 issue 讨论你想要改变的内容。

## 许可证

[MIT License](LICENSE)

---

# GraphRAG4OpenWebUI

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

[MIT License](LICENSE)
