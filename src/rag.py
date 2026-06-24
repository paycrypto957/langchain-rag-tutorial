import asyncio

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from cacheRephraseRetrieve import rephrase_retrieve, get_rag_chain, get_llm

# 存储对话历史
chat_history = []

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
    answer = ""

    input = {"query": query, "history": chat_history}

    # 1、获取检索器
    # retriever = get_retriever(k=20, embedding_model=embedding_model)
    # 2、执行重述、检索
    retrieve_result = rephrase_retrieve(input, llm)
    # 3、获取RAG链
    rag_chain = get_rag_chain(retrieve_result, llm)
    # 4、异步执行RAG链，流式输出
    async for chunk in rag_chain.astream(input):
        answer += chunk
        yield chunk  # 将大模型生成的内容逐块(chunk)地返回给调用者，而不是等待整个回答完成后一次性返回

    # 5、更新对话历史，添加用户查询和AI回答
    chat_history.append(
        {"role": "user", "content": query, "conversation_id": conversation_id}
    )
    chat_history.append(
        {"role": "ai", "content": answer, "conversation_id": conversation_id}
    )


if __name__ == '__main__':
    async def main():
        query_list = ["不动产或者动产被人占有怎么办", "那要是被损毁了呢"]
        for query in query_list:
            print(f"===== 查询: {query} =====")
            async for chunk in invoke_rag(query, 1, chat_history):
                print(chunk, end="", flush=True)


    asyncio.run(main())
