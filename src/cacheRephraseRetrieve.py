import functools
import os
from collections import OrderedDict
from datetime import datetime
from typing import Dict

import faiss
import numpy as np
import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.llms.tongyi import Tongyi
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


def format_history(history, max_epoch=3):
    # 每轮对话有 用户问题 和 助手回复
    if len(history) > 2 * max_epoch:
        history = history[-2 * max_epoch:]
    return "\n".join([f"{i['role']}：{i['content']}" for i in history])


def format_docs(docs: list[Document]) -> str:
    """格式化 docs"""
    return "\n\n".join(doc.page_content for doc in docs)


# ------------------ 缓存优化1：对最近的query的Embedding结果缓存到LRU中 ------------------
# 1、初始化Embedding模型
# embedding_mode为HuggingFaceEmbeddings，不可哈希，不能作为缓存的形参，因此挪到该文件作为全局变量
embedding_model = HuggingFaceEmbeddings(
    model_name="../bge-base-zh-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={
        "normalize_embeddings": True
    },  # 输出归一化向量，更适合余弦相似度计算
)


# LRU 缓存：缓存最近1024个查询的嵌入结果
@functools.lru_cache(maxsize=1024)
def embed(query: str) -> list[float]:
    """文本嵌入"""
    return embedding_model.embed_query(query)


# ------------缓存优化2：对检索结果缓存 -----------------------------
class RetrievalCache:
    """检索结果缓存"""

    def __init__(
            self,
            embedding_dim: int,
            max_cache_entries: int = 512,
            similarity_threshold: float = 0.9,
            ttl: int = 3600,
    ):
        self.max_cache_entries = max_cache_entries  # 最大缓存条目数
        self.similarity_threshold = similarity_threshold  # 相似度阈值
        self.ttl = ttl  # 老化时间

        # 缓存字典 {faiss_index_id : {'docs': [Document], 'timestamp': datetime}}
        # self.cache: OrderedDict[int, dict[str,any]] = OrderedDict()
        self.cache = OrderedDict()
        # 创建 FAISS 索引，存储查询嵌入向量
        self.query_vectorstore = faiss.IndexFlatL2(embedding_dim)
        # 用于 FAISS 索引的内部 ID， ID 通常是添加的顺序
        self.current_index_id = 0

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(embedding: list[float], k: int = 20) -> list[Document]:
            # 将嵌入向量转换为 numpy 数组
            query_embedding_np = np.array(embedding).astype(np.float32)

            # 惰性清理
            self._lazy_evict()

            # 如果缓存不为0，则在 FAISS 中查找相似的查询嵌入
            if self.query_vectorstore.ntotal > 0:
                # 搜索最相似的 1 个向量
                distances, indices = self.query_vectorstore.search(
                    query_embedding_np.reshape(1, -1), 1
                )

                # 得到该向量与查询向量的 L2 距离的平方，以及对应的 ID
                distance = distances[0, 0]
                cached_id = indices[0, 0]

                # 我们认定如果 l2_distance <= (1 - similarity_threshold)^2 则足够相似
                if (
                        cached_id != -1
                        and distance <= (1 - self.similarity_threshold) ** 2
                        and cached_id in self.cache
                ):
                    # 将命中的项移动到队尾
                    self.cache.move_to_end(cached_id)
                    # 更新被命中的缓存的时间戳
                    self.cache[cached_id]["timestamp"] = datetime.now()
                    # 返回命中的结果
                    return self.cache[cached_id]["docs"]

            # 如果未命中缓存，执行检索，并将结果存入缓存
            retrieved_docs = func(embedding, k)
            self._add_to_cache(query_embedding_np, retrieved_docs)

            return retrieved_docs

        return wrapper

    def _add_to_cache(
            self, query_embedding: np.ndarray, retrieved_docs: list[Document]
    ):
        """将查询嵌入和检索结果添加到缓存"""
        self._lazy_evict()
        # 将新的嵌入添加到 FAISS
        self.query_vectorstore.add(query_embedding.reshape(1, -1))
        # 缓存检索结果
        self.cache[self.current_index_id] = {
            "docs": retrieved_docs,
            "timestamp": datetime.now(),
        }
        self.current_index_id += 1

    def _lazy_evict(self):
        """惰性清理"""
        curremt_time = datetime.now()
        removed_entries = []

        # 清理过期数据
        for faiss_index_id, entry in list(self.cache.items()):
            if (curremt_time - entry["timestamp"]).total_seconds() < self.ttl:
                break
            del self.cache[faiss_index_id]
            removed_entries.append(faiss_index_id)

        # 清理超出容量的数据
        while len(self.cache) > self.max_cache_entries:
            removed_entries.append(self.cache.popitem(last=False))

        # 从 FAISS 中清理数据
        if removed_entries:
            self.query_vectorstore.remove_ids(
                np.array(removed_entries).astype(np.int64)
            )

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        # 重新初始化 FAISS 索引
        self.query_vectorstore = faiss.IndexFlatL2(self.query_vectorstore.d)
        self.current_index_id = 0


# ------------缓存优化2：对检索结果缓存 -----------------------------
# TTL 缓存：缓存 1 小时内的 512 条检索结果
@RetrievalCache(embedding_dim=768)
def retrieve(embedding: list[float], k=20) -> list[Document]:
    """向量检索"""
    vectorstore = Chroma(persist_directory="vectorstore")
    docs = vectorstore.similarity_search_by_vector(embedding, k=k)
    return docs


def get_llm():
    # 大模型
    load_dotenv()
    TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
    llm = Tongyi(model="qwen-turbo", api_key=TONGYI_API_KEY)
    return llm


def rephrase_retrieve(input: Dict[str, str], llm):
    """重述用户query，检索向量数据库"""

    # 1、重述query的prompt
    rephrase_prompt = PromptTemplate.from_template(
        """
        根据对话历史简要完善最新的用户消息，使其更加具体。只输出完善后的问题。如果问题不需要完善，请直接输出原始问题。
        
        {history}
        用户：{query}
        """
    )

    # 2、重述链条：根据历史和当前 query 生成更具体问题
    rephrase_chain = (
            {
                "history": lambda x: format_history(x.get("history")),
                "query": lambda x: x.get("query"),
            }
            | rephrase_prompt
            | llm
            | StrOutputParser()
            | (lambda x: print(f"===== 重述后的查询: {x}=====") or x)
    )

    # 3、执行重述
    rephrase_query = rephrase_chain.invoke({"history": input.get("history"), "query": input.get("query")})

    # 4、使用重述后的query进行检索
    # 这里应用缓存，所以拆分成 Embedding 和 Retrieve分别执行
    embeddings = embed(rephrase_query)  # 会应用缓存1：对最近的query的Embedding结果缓存到LRU中
    retrieve_result = retrieve(embeddings)  # 会应用缓存2：对检索结果缓存

    return retrieve_result


def get_rag_chain(retrieve_result, llm):
    """使用检索结果、历史记录、用户查询，提交大模型生成回复"""

    # 1、Prompt 模板
    prompt = PromptTemplate(
        input_variables=["context", "history", "query"],
        template="""
                      你是一个专业的中文问答助手，擅长基于提供的资料回答问题。
                      请仅根据以下背景资料以及历史消息回答问题，如无法找到答案，请直接回答“我不知道”。
                      
                      背景资料：{context}
                      
                      历史消息：[{history}]
                      
                      问题：{query}
                      
                      回答：""",
    )

    # 2、定义 RAG 链条
    rag_chain = (
            {
                "context": lambda x: format_docs(retrieve_result),
                "history": lambda x: format_history(x.get("history")),
                "query": lambda x: x.get("query"),
            }
            | prompt
            | (lambda x: print(x.text, end="") or x)
            | llm
            | StrOutputParser()  # 输出解析器，将输出解析为字符串
    )

    return rag_chain
