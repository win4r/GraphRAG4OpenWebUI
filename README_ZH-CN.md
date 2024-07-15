# GraphRAG4OpenWebUI

<div align="center">
  <p><strong>将微软的 GraphRAG 技术集成到 Open WebUI 中，实现高级信息检索</strong></p>
  <a href="README.md">English</a> | 简体中文
</div>

GraphRAG4OpenWebUI 是一个专为 Open WebUI 设计的 API 接口，旨在集成微软研究院的 GraphRAG（基于图的检索增强生成）技术。该项目提供了一个强大的信息检索系统，支持多种搜索模型，特别适合在开放式 Web 用户界面中使用。

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

确保您的系统中已安装 Python 3.8 或更高版本。然后，按照以下步骤安装：

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

在运行 API 之前，需要设置以下环境变量。您可以通过创建 `.env` 文件或直接在终端中导出这些变量：

```bash
export GRAPHRAG_API_KEY="your_graphrag_api_key"
export TAVILY_API_KEY="your_tavily_api_key"
export GRAPHRAG_LLM_MODEL="gpt-3.5-turbo"
export GRAPHRAG_EMBEDDING_MODEL="text-embedding-3-small"
export INPUT_DIR="/path/to/your/input/directory"
```

请确保将上述命令中的占位符替换为实际的 API 密钥和路径。

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
       "messages": [{"role": "user", "content": "您的搜索查询"}],
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
- API 使用异步编程，确保您的环境支持异步操作。
- 对于大规模部署，建议使用生产级的 ASGI 服务器。
- 本项目专为 Open WebUI 设计，可以轻松集成到各种基于 Web 的应用中。

## 贡献

我们欢迎您提交 Pull Requests 来改进这个项目。对于重大变更，请先开 issue 讨论您想要改变的内容。

## 许可证

[Apache-2.0 许可证](LICENSE)
