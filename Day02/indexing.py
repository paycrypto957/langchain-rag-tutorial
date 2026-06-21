import json
import re

import torch
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
      TextLoader,
      UnstructuredMarkdownLoader,
      UnstructuredPDFLoader,
      UnstructuredWordDocumentLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents():
    """
    加载多种类型的文档，包括text、markdown、PDF和Word文档

    Returns:
          list: 包含所有加载文档的列表
    """
    # 加载文本文件
    text_documents = TextLoader(
        "../knowledge_base/sample.txt",
        encoding="utf8"
    ).load()

    # 加载Markdown文件
    md_documents = UnstructuredMarkdownLoader(
        "../knowledge_base/sample.md"
    ).load()

    # 加载PDF文件
    pdf_documents = UnstructuredPDFLoader(
        "../knowledge_base/sample.pdf",
        mode="elements",  # 元素模式
        strategy="hi_res",  # 高分辨率策略
        # strategy="fast",
        languages=["eng", "chi_sim"],  # 支持的语言：英文和简体中文
    ).load()

    # 加载Word文档
    word_documents = UnstructuredWordDocumentLoader(
        "../knowledge_base/sample.docx"
    ).load()

    # 返回所有文档的列表
    return text_documents + md_documents + pdf_documents + word_documents


def clean_content(documents: list):
    """文本清洗"""
    cleaned_docs = []

    for doc in documents:

        # 1、page_content处理：去除多余换行和空格
        text = doc.page_content

        # 将连续的换行符替换为两个换行符，正则表达式模式：r"\n{2,}"
        # r"" 表示原始字符串（raw string），避免转义字符的特殊处理
        # \n 表示换行符
        # {2,} 是量词，表示前面的字符（换行符）出现 2 次或更多次
        text = re.sub(r"\n{2,}", "\n\n", text)
        text = text.strip()

        # 2、metadata处理：将所有非 Chroma 支持类型的值转为 JSON 格式字符串
        allowed_types = (str, int, float, bool)
        for key, value in doc.metadata.items():
            if not isinstance(value, allowed_types):
                try:
                    doc.metadata[key] = json.dumps(value, ensure_ascii=False)
                except (TypeError, ValueError):
                    # 如果 json.dumps 失败（如包含不可序列化对象），转为 str
                    doc.metadata[key] = str(value)

        # 3、更新文档内容
        doc.page_content = text
        cleaned_docs.append(doc)

    return cleaned_docs


def text_split(documents):
    # 文本分块
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "。"],  # 分隔符列表
        chunk_size=400,  # 每个块的最大长度
        chunk_overlap=40,  # 每个块重叠的长度
    )
    texts = text_splitter.split_documents(documents)
    return texts


def save_to_db(texts):
    # 加载嵌入模型
    embedding_model = HuggingFaceEmbeddings(
        model_name="../bge-base-zh-v1.5",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={
            "normalize_embeddings": True
        },  # 输出归一化向量，更适合余弦相似度计算
    )
    # 嵌入并存储到向量数据库
    vectorstore = Chroma.from_documents(
        texts,  # 文档列表
        embedding_model,  # 嵌入模型
        persist_directory="vectorstore",  # 存储路径
    )
    return vectorstore


if __name__ == '__main__':
    # 1. 加载文档
    documents = load_documents()
    # 2、清洗文档
    cleaned_docs = clean_content(documents)
    # 3、切分文档
    texts = text_split(cleaned_docs)
    # 4、保存到数据库中
    vectorstore = save_to_db(texts)
    # 5、查看数据库内容
    print(vectorstore.get().keys())  # 查看所有属性
    print(vectorstore.get(include=["embeddings"])["embeddings"][:5, :5])  # 查看嵌入向量
