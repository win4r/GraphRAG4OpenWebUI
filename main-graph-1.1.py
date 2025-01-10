import os
import asyncio
import time
import uuid
import json
import re
import pandas as pd
import tiktoken
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
from tavily import TavilyClient
from colorama import init, Fore
# 初始化 colorama
init(autoreset=True)
# GraphRAG 相关导入
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_communities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_report_embeddings
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.drift_search.drift_context import (
    DRIFTSearchContextBuilder,
)
from graphrag.config.models.drift_search_config import DRIFTSearchConfig
from graphrag.query.structured_search.drift_search.search import DRIFTSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
# Athene-V2-Chat_exl2_2.25bpw，Rombos-Coder-V2.5-Qwen-32b-exl2_5.0bpw
LLM_MODEL = "Rombos-LLM-V2.5-Qwen-32b-4.5bpw-exl2"

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置常量和配置
INPUT_DIR = "E:\\graphrag_kb\\input\\artifacts"
LANCEDB_URI = "E:\\graphrag_kb\\output\\lancedb"
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
FINAL_COMMUNITY_TABLE = "create_final_communities"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_LEVEL = 10     # 社区层级，越高表示使用更精细的社区报告（但计算成本更高），默认2
PORT = 8013

# 全局变量，用于存储搜索引擎和问题生成器
local_search_engine = None
global_search_engine = None
drift_serch_engine = None
question_generator = None


# 数据模型
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 0.7
    n: Optional[int] = 1
    stream: Optional[bool] = True
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 12_000
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage
    system_fingerprint: Optional[str] = None


async def setup_llm_and_embedder():
    """
    设置语言模型（LLM）和嵌入模型
    知识图谱无法正常使用时用gemini-1.5-flash-latest模型重新训练小文档生成图谱
    """
    """
    # 获取API密钥和基础URL
    api_key = "xxx"
    api_key_embedding = "xxx"
    api_base = "https://ai.liaobots.work/v1"
    api_base_embedding = "https://ai.liaobots.work/v1"
    api_base = "http://localhost:11434/v1"
    # 获取模型名称
    LLM_MODEL = "gpt-4o-mini"
    embedding_model = "text-embedding-ada-002"
    """
    logger.info("正在设置LLM和嵌入器")
    # ollama获取API密钥和基础URL
    api_key = "xxx"
    api_base = "http://127.0.0.1:5001/v1"
    global LLM_MODEL
    logger.info(Fore.CYAN + f"GRAPHRAG使用模型：{LLM_MODEL}")
    # 初始化ChatOpenAI实例
    llm = ChatOpenAI(
        api_key=api_key,
        api_base=api_base,
        model=LLM_MODEL,
        api_type=OpenaiApiType.OpenAI,
        max_retries=10,
        request_timeout=120  # 设置超时时间为120秒
    )

    # 初始化token编码器
    token_encoder = tiktoken.get_encoding("cl100k_base")

    # 初始化文本嵌入模型
    # openai在线模型
    """
    api_key="sk-9mxwRPHwHt8M1ct7CaCf041d6fC44e9587A041Ca3145E09e",
    api_base="https://apis.wumingai.com/v1",
    model=embedding_model,
    deployment_name=embedding_model,
    """
    # xinference本地嵌入模型
    """
    api_key="xinference",
    api_base="http://127.0.0.1:9997/v1",
    model="bge-m3",
    deployment_name="bge-m3",
    """
    text_embedder = OpenAIEmbedding(
        # 本地嵌入模型
        api_key="ollama",
        api_base="http://localhost:11434/v1",
        model="bge-m3:Q4",
        deployment_name="bge-m3:Q4",
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )

    logger.info("LLM和嵌入器设置完成")
    return llm, token_encoder, text_embedder

def embed_text(column):
    text_embedder = OpenAIEmbedding(
        # 本地嵌入模型
        api_key="ollama",
        api_base="http://localhost:11434/v1",
        model="bge-m3:Q4",
        deployment_name="bge-m3:Q4",
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )

    return column.apply(lambda x: text_embedder.embed(x))
async def load_context():
    """
    加载上下文数据，包括实体、关系、报告、文本单元和协变量
    """
    logger.info("正在加载上下文数据")
    try:
        entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
        entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
        entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

        description_embedding_store = LanceDBVectorStore(collection_name="default-entity-description")
        description_embedding_store.connect(db_uri=LANCEDB_URI)

        relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
        relationships = read_indexer_relationships(relationship_df)

        report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
        #reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL,content_embedding_col="full_content_embeddings")
        reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL,content_embedding_col="full_content_embeddings")
        full_content_embedding_store = LanceDBVectorStore(collection_name="default-community-full_content")
        full_content_embedding_store.connect(db_uri=LANCEDB_URI)
        read_indexer_report_embeddings(reports, full_content_embedding_store)
        
        final_communities_df = pd.read_parquet(f"{INPUT_DIR}/{FINAL_COMMUNITY_TABLE}.parquet")
        communities = read_indexer_communities(final_communities_df,entity_df,report_df)


        text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
        text_units = read_indexer_text_units(text_unit_df)
        covariate_df = pd.read_parquet(f"{INPUT_DIR}/{COVARIATE_TABLE}.parquet")
        claims = read_indexer_covariates(covariate_df)
        logger.info(f"声明记录数: {len(claims)}")
        covariates = {"claims": claims}

        logger.info("上下文数据加载完成")
        return entities, relationships, reports, communities,text_units, description_embedding_store, covariates
    except Exception as e:
        logger.error(f"加载上下文数据时出错: {str(e)}")
        raise


async def setup_search_engines(llm, token_encoder, text_embedder, entities, relationships, reports,communities, text_units,
                            description_embedding_store, covariates):
    """
    设置本地搜索引擎和全局搜索引擎
    """
    logger.info("正在设置搜索引擎")

    # 设置本地搜索引擎
    local_context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=covariates,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": 12_000,
    }

    local_llm_params = {
        "max_tokens": 12_000,
        "temperature": 0.3,
    }

    local_search_engine = LocalSearch(
        llm=llm,
        context_builder=local_context_builder,
        token_encoder=token_encoder,
        llm_params=local_llm_params,
        context_builder_params=local_context_params,
        response_type="multiple paragraphs",
    )

    # 设置全局搜索引擎
    global_context_builder = GlobalCommunityContext(
        communities = communities ,
        community_reports=reports,
        entities=entities,
        token_encoder=token_encoder,
    )

    global_context_builder_params = {
        "use_community_summary": False,
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0.5,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 12_000,
        "temperature": 0.5,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 12_000,
        "temperature": 0.5,
    }

    global_search_engine = GlobalSearch(
        llm=llm,
        context_builder=global_context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=True,
        json_mode=True,
        context_builder_params=global_context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",
    )
    drift_params = DRIFTSearchConfig()
    drift_params.temperature = 0.5
    drift_params.max_tokens = 12_000
    drift_context_builder = DRIFTSearchContextBuilder(
            chat_llm=llm,
            text_embedder=text_embedder,
            entities=entities,
            relationships=relationships,
            reports=reports,
            entity_text_embeddings=description_embedding_store,
            text_units=text_units,
            config = drift_params

        )
    drift_serch_engine = DRIFTSearch(
            llm=llm, context_builder=drift_context_builder, token_encoder=token_encoder
        )
    logger.info("搜索引擎设置完成")
    return local_search_engine, global_search_engine,drift_serch_engine, local_context_builder, local_llm_params, local_context_params


def format_response(response):
    """
    格式化响应，添加适当的换行和段落分隔。
    """
    modified_text = re.sub(r"`(.*?)`", r'```python\1```', response)

    return modified_text


async def tavily_search(prompt: str):
    """
    使用Tavily API进行搜索
    """
    try:
        client = TavilyClient(api_key=os.environ['TAVILY_API_KEY'])
        resp = client.search(prompt, search_depth="advanced")

        # 将Tavily响应转换为Markdown格式
        markdown_response = "# 搜索结果\n\n"
        for result in resp.get('results', []):
            markdown_response += f"## [{result['title']}]({result['url']})\n\n"
            markdown_response += f"{result['content']}\n\n"

        return markdown_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tavily搜索错误: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    global local_search_engine, global_search_engine,drift_serch_engine, question_generator
    try:
        logger.info("正在初始化搜索引擎和问题生成器...")
        llm, token_encoder, text_embedder = await setup_llm_and_embedder()
        entities, relationships, reports,communities, text_units, description_embedding_store, covariates = await load_context()
        local_search_engine, global_search_engine,drift_serch_engine, local_context_builder, local_llm_params, local_context_params = await setup_search_engines(
            llm, token_encoder, text_embedder, entities, relationships, reports,communities, text_units,
            description_embedding_store, covariates
        )

        question_generator = LocalQuestionGen(
            llm=llm,
            context_builder=local_context_builder,
            token_encoder=token_encoder,
            llm_params=local_llm_params,
            context_builder_params=local_context_params,
        )
        logger.info("初始化完成。")
    except Exception as e:
        logger.error(f"初始化过程中出错: {str(e)}")
        raise

    yield

    # 关闭时执行
    logger.info("正在关闭...")


app = FastAPI(lifespan=lifespan)


# 在 chat_completions 函数中添加以下代码

async def full_model_search(prompt: str):
    """
    执行全模型搜索，包括本地检索、全局检索和 Tavily 搜索
    """
    local_result = await local_search_engine.asearch(prompt)
    global_result = await global_search_engine.asearch(prompt)
    drift_result = await drift_serch_engine.asearch(prompt)
    tavily_result = await tavily_search(prompt)

    # 格式化结果
    formatted_result = "# 🔥🔥🔥综合搜索结果\n\n"

    formatted_result += "## 🔥🔥🔥本地检索结果\n"
    formatted_result += local_result.response + "\n\n"

    formatted_result += "## 🔥🔥🔥全局检索结果\n"
    formatted_result += global_result.response + "\n\n"

    formatted_result += "## 🔥🔥🔥混合检索结果\n"
    formatted_result += drift_result.response + "\n\n"

    formatted_result += "## 🔥🔥🔥Tavily 搜索结果\n"
    formatted_result += tavily_result + "\n\n"

    return formatted_result

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not local_search_engine or not global_search_engine or not drift_serch_engine:
        logger.error("搜索引擎未初始化")
        raise HTTPException(status_code=500, detail="搜索引擎未初始化")

    prompt = request.messages[-1].content
    logger.info(Fore.CYAN + f"收到模型请求内容：{prompt}")
    # 根据模型选择使用不同的搜索方法
    if request.model == "graphrag-global-search:latest":
        result = await global_search_engine.asearch(prompt)
        formatted_response = result.response
    elif request.model == "graphrag-drift-search:latest":
        result = await drift_serch_engine.asearch(prompt)
        formatted_response = result.response
        formatted_response:str = formatted_response["nodes"][0]["answer"]
        formatted_response = formatted_response.replace(" n n","\n")
    elif request.model == "tavily-search:latest":
        result = await tavily_search(prompt)
        formatted_response = result
    elif request.model == "full-model:latest":
        formatted_response = await full_model_search(prompt)
    else:  # 默认使用本地搜索
        result = await local_search_engine.asearch(prompt)
        # 格式化回复
        #formatted_response = format_response(result.response)
        formatted_response = result.response

    logger.info(Fore.CYAN + f"知识图谱的搜索结果: {formatted_response}")
    # 流式响应和非流式响应的处理保持不变
    if request.stream:
        async def generate_stream():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            lines = formatted_response.split('\n')
            for i, line in enumerate(lines):
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            #"delta": {"content": line + '\n'} if i > 0 else {"role": "assistant", "content": ""},
                            "delta": {"content": line + '\n'}, # if i > 0 else {"role": "assistant", "content": ""},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.05)

            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    else:
        response = ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=formatted_response),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(formatted_response.split()),
                total_tokens=len(prompt.split()) + len(formatted_response.split())
            )
        )
        logger.info(f"发送响应: {response}")
        return JSONResponse(content=response.dict())

@app.get("/v1/models")
async def list_models():
    """
    返回可用模型列表
    """
    logger.info("收到模型列表请求")
    current_time = int(time.time())
    models = [
        {"id": "graphrag-local-search:latest", "object": "model", "created": current_time - 100000, "owned_by": "graphrag"},
        {"id": "graphrag-global-search:latest", "object": "model", "created": current_time - 95000, "owned_by": "graphrag"},
        {"id": "graphrag-drift-search:latest", "object": "model", "created": current_time - 90000, "owned_by": "graphrag"},
        # {"id": "graphrag-question-generator:latest", "object": "model", "created": current_time - 90000, "owned_by": "graphrag"},
        # {"id": "gpt-3.5-turbo:latest", "object": "model", "created": current_time - 8_0000, "owned_by": "openai"},
        # {"id": "text-embedding-3-small:latest", "object": "model", "created": current_time - 70000, "owned_by": "openai"},
        #{"id": "tavily-search:latest", "object": "model", "created": current_time - 85000, "owned_by": "tavily"},
        # {"id": "full-model:latest", "object": "model", "created": current_time - 8_0000, "owned_by": "combined"}

    ]

    response = {
        "object": "list",
        "data": models
    }

    logger.info(f"发送模型列表: {response}")
    return JSONResponse(content=response)

if __name__ == "__main__":
    import uvicorn

    logger.info(f"在端口 {PORT} 上启动服务器")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

