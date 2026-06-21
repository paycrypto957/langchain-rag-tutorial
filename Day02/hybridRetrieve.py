import os
from hashlib import sha256
from typing import Dict

import jieba
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.llms.tongyi import Tongyi
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


def format_history(history, max_epoch=3):
    # 每轮对话有 用户问题 和 助手回复
    if len(history) > 2 * max_epoch:
        history = history[-2 * max_epoch:]
    return "\n".join([f"{i['role']}：{i['content']}" for i in history])


def format_docs(docs: list[Document]) -> str:
    """格式化 docs"""
    return "\n\n".join(doc.page_content for doc in docs)


def get_retriever(k=20, embedding_model=None):
    """获取向量数据库的检索器"""

    # 1、初始化 Chroma 客户端
    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=embedding_model,
    )

    # 2、创建向量数据库检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # 检索方式，similarity 或 mmr
        search_kwargs={"k": k},
    )

    return retriever


# -------------检索后优化：混合检索--------------------
def get_bm25_retriever(k=20):
    # 1、初始化 Chroma 客户端
    vectorstore = Chroma(persist_directory="vectorstore")
    # 2、BM25检索器
    all_docs = vectorstore.get()
    bm25_retriever = BM25Retriever.from_documents(
        [
            Document(id=doc_id, page_content=doc, metadata=meta)
            for doc_id, doc, meta in zip(
            all_docs["ids"], all_docs["documents"], all_docs["metadatas"]
        )
        ],
        preprocess_func=lambda x: jieba.lcut(x),
    )
    bm25_retriever.k = k
    return bm25_retriever


def get_llm():
    # 大模型
    load_dotenv()
    TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
    llm = Tongyi(model="qwen-turbo", api_key=TONGYI_API_KEY)
    return llm


# --------- rrf重排序 ---------
def reciprocal_rank_fusion(
        docs: list[list[Document]], k=60, docs_return_num=10
) -> list[Document]:
    """
    倒数排序融合：将多个排序列表融合为一个最终的排序列表，优先考虑在多个列表中排名靠前的文档
    对多查询返回的多个文档列表进行排序，返回一个排序后的文档列表
    """
    # 初始化每个文档的融合分数
    fused_scores = {}
    # 用于存储去重后的文档，键是内容的哈希值，值是原始的 Document 对象
    unique_docs_by_content = {}

    for doc_list in docs:
        # 遍历每个有序列表
        for rank, doc in enumerate(doc_list):
            key = sha256(doc.page_content.encode("utf-8")).hexdigest()
            unique_docs_by_content[key] = doc
            fused_scores[key] = fused_scores.get(key, 0) + 1 / (rank + k)
    # 按融合分数排序
    sorted_content_hashes = sorted(
        fused_scores.items(), key=lambda x: x[1], reverse=True
    )
    # 根据排序后的内容哈希值，从 unique_docs_by_content 中取出原始的 Document 对象
    reranked_docs = []
    for key, _ in sorted_content_hashes:
        reranked_docs.append(unique_docs_by_content[key])

    return reranked_docs[:docs_return_num]


def rephrase_retrieve(input: Dict[str, str], llm, retriever, bm25_retriever):
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
    retrieve_result = retriever.invoke(rephrase_query, k=3)

    # -----------------检索后优化：混合检索 ---------
    # 5、使用BM25进行检索
    bm25_retrieve_result = bm25_retriever.invoke(rephrase_query, k=3)

    # 6、对两种检索结果进行RRF重排序
    result = reciprocal_rank_fusion([retrieve_result, bm25_retrieve_result], k=60, docs_return_num=3)

    return result


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
