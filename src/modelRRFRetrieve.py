import os
from typing import Dict

import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.llms.tongyi import Tongyi
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


def get_llm():
    # 大模型
    load_dotenv()
    TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
    llm = Tongyi(model="qwen-turbo", api_key=TONGYI_API_KEY)
    return llm


# --------- 检索后优化：使用模型重排序 ---------
def model_rerank(
        query, docs: list[Document], rerank_tokenizer, rerank_model, docs_return_num=10, batch_size=64,
) -> list[Document]:
    """模型重排序：使用模型对文档进行重排序"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scores = []
    # 逐批次处理
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i: i + batch_size]
        inputs = rerank_tokenizer(
            text=[query] * len(batch_docs),
            text_pair=[doc.page_content for doc in batch_docs],
            padding=True,
            max_length=rerank_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        # 计算相关性分数
        with torch.no_grad():
            outputs = rerank_model(**inputs)
            batch_scores = outputs.logits.squeeze()
        scores.extend(batch_scores.tolist())

    # 按相关性分数排序
    reranked_docs = [
        doc for doc, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    ]

    return reranked_docs[:docs_return_num]


def rephrase_retrieve(input: Dict[str, str], llm, retriever, rerank_tokenizer, rerank_model):
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

    # 4、使用重述后的query进行检测
    retrieve_result = retriever.invoke(rephrase_query, k=3)

    # ----------------- 检索后优化：使用模型重排序 ---------
    # 5、使用模型重排序
    rerank_result = model_rerank(input.get("query"), retrieve_result, rerank_tokenizer=rerank_tokenizer,
                                 rerank_model=rerank_model)
    return rerank_result


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
