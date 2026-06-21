import asyncio

import torch
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import ContextRelevance, answer_relevancy, faithfulness, ResponseGroundedness
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rephraseRetrieve import rephrase_retrieve, get_rag_chain, get_llm, get_retriever

# 存储对话历史
chat_history = []
# 存储评估用的检索结果
retrieve_history = []

# 1、初始化Embedding模型
embedding_model = HuggingFaceEmbeddings(
    model_name="../bge-base-zh-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={
        "normalize_embeddings": True
    },  # 输出归一化向量，更适合余弦相似度计算
)

# 2、初始化 LLM
llm = get_llm()


async def invoke_rag(query, conversation_id, chat_history):
    rerank_model = AutoModelForSequenceClassification.from_pretrained("../bge-reranker-base")
    rerank_tokenizer = AutoTokenizer.from_pretrained("../bge-reranker-base")

    answer = ""

    input = {"query": query, "history": chat_history}

    # 1、获取检索器
    retriever = get_retriever(k=20, embedding_model=embedding_model)
    # bm25_retriever = get_bm25_retriever() #配合混合检索 hy
    # 2、执行重述、检索
    retrieve_result = rephrase_retrieve(input, llm, retriever)
    # 3、获取RAG链
    rag_chain = get_rag_chain(retrieve_result, llm)
    # 4、异步执行RAG链，流式输出
    async for chunk in rag_chain.astream(input):
        answer += chunk
        yield chunk

    # 5、更新对话历史，添加用户查询和AI回答
    chat_history.append(
        {"role": "user", "content": query, "conversation_id": conversation_id}
    )
    chat_history.append(
        {"role": "ai", "content": answer, "conversation_id": conversation_id}
    )

    # 6、保存检索的结果，用于后续评估
    retrieve_history.append({
        "query": query,
        "answer": answer,
        "contexts": [
            doc.page_content
            for doc in retrieve_result
        ]
    })


async def rag_evaluate(datas):
    """
    评估RAG模型性能
    使用RAGAS框架评估RAG系统的各项指标

    Args:
              datas (list): 评估数据列表，包含查询、回答和上下文信息

    Returns:
              EvaluationResult: 评估结果对象
    """

    # 1、构建RAGAS评估数据集
    ragas_data = {
        "user_input": [d["query"] for d in datas],  # 用户查询
        "response": [d["answer"] for d in datas],  # AI回答
        "retrieved_contexts": [d["contexts"] for d in datas],  # 检索到的上下文
    }
    dataset = Dataset.from_dict(ragas_data)

    # 2、定义评估指标
    metrics = [
        ContextRelevance(),  # 上下文相关性
        answer_relevancy,  # 答案相关性
        faithfulness,  # 忠实度
        ResponseGroundedness(),  # 响应真实性
    ]

    # 3、执行评估
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,  # 使用的大语言模型
        embeddings=embedding_model,  # 使用的嵌入模型
        raise_exceptions=False,  # 允许在评估失败时返回 NaN 而不是抛出异常
    )

    # 4、清空保存的检索结果
    datas.clear()

    return result


if __name__ == '__main__':
    async def main():

        query_list = ["不动产或者动产被人占有怎么办", "那要是被损毁了呢"]
        for query in query_list:
            print(f"===== 查询: {query} =====")
            async for chunk in invoke_rag(query, 1, chat_history):
                print(chunk, end="", flush=True)

        # 评估模型性能
        evaluate_res = await rag_evaluate(retrieve_history)
        print(evaluate_res)
        # 输出评估结果的关键指标
        import pandas as pd
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(
            evaluate_res.to_pandas()[
                [
                    "nv_context_relevance",
                    "answer_relevancy",
                    "faithfulness",
                    "nv_response_groundedness",
                ]
            ]
        )


    asyncio.run(main())
