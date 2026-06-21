# LangChain RAG Tutorial

一个从零开始的 LangChain RAG 实战教程,按 Day01 → Day02 由浅入深,从最基础的文档加载、LCEL 链,一路写到流式问答 Web App 和 RAGAS 评估。

- **Embedding**:`bge-base-zh-v1.5`(本地)
- **LLM**:阿里通义千问 `qwen-turbo`(DashScope)
- **Vector Store**:Chroma
- **语料**:`knowledge_base/` 下的 txt / md / pdf / docx 示例文档

## 目录结构

```
.
├── Day01/                      # 基础入门 (notebooks)
│   ├── TestLoadDoc.ipynb       # TextLoader 加载单个 txt
│   ├── TestLoadAllDoc.ipynb    # 多格式加载器 (txt/md/pdf/docx)
│   ├── TestRetrieve.ipynb      # bge embedding + Chroma 相似度检索
│   ├── TestLCEL01.ipynb        # LCEL 入门: prompt | LLM | parser
│   ├── TestLCEL02.ipynb        # LCEL + 对话历史
│   └── TestLCEL03.ipynb        # LCEL 进阶组合
│
├── Day02/                      # 进阶实战 (模块化 Python)
│   ├── indexing.py             # 文档入库: 加载/清洗/切分(400/40)/持久化
│   ├── rag.py                  # 异步流式 RAG 入口
│   ├── rephraseRetrieve.py     # 基于对话历史的 query 重写
│   ├── cacheRephraseRetrieve.py# LRU + 语义缓存 (FAISS/SHA256)
│   ├── HyDERetrieve.py         # HyDE: 生成假设答案再检索
│   ├── multiRetrieve.py        # 多查询检索 (LLM 生成子查询)
│   ├── hybridRetrieve.py       # 混合检索: Dense (Chroma) + Sparse (BM25/jieba)
│   ├── RRFRetrieve.py          # Reciprocal Rank Fusion
│   ├── modelRRFRetrieve.py     # RRF + CrossEncoder (bge-reranker-base) 重排
│   ├── metric.py               # RAGAS 评估 (faithfulness / relevancy / ...)
│   ├── app.py                  # FastAPI + SSE 流式问答 Web App (127.0.0.1:8089)
│   └── templates/
│       └── naive_index.html    # 前端聊天 UI
│
├── knowledge_base/             # 示例语料
│   ├── sample.txt
│   ├── sample.md
│   ├── sample.pdf
│   └── sample.docx
│
├── bge-base-zh-v1.5/           # 本地 embedding 模型 (gitignored)
└── bge-reranker-base/          # 本地 reranker 模型 (gitignored)
```

## 快速开始

### 1. 环境要求

- Python 3.10+(代码用到 `list[Document]` 等 PEP 585 语法)
- 依赖(无 requirements.txt,按需安装):

```bash
pip install langchain langchain-chroma langchain-huggingface langchain-community \
            chromadb faiss-cpu jieba torch transformers datasets ragas \
            unstructured fastapi uvicorn python-dotenv
```

### 2. 下载本地模型

仓库未包含模型权重(GitHub 单文件 100MB 限制),请从 HuggingFace 拉取放到项目根目录:

```bash
# embedding 模型
git clone https://huggingface.co/BAAI/bge-base-zh-v1.5

# reranker 模型(仅 modelRRFRetrieve.py 需要)
git clone https://huggingface.co/BAAI/bge-reranker-base
```

或使用 modelscope 镜像。

### 3. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env,填入 DashScope API Key
```

`.env` 示例:

```
TONGYI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> 申请地址:https://dashscope.console.aliyun.com/

### 4. 构建索引

```bash
cd Day02
python indexing.py
```

会读取 `knowledge_base/` 下的文档,切分后写入 `Day02/vectorstore/`。

### 5. 启动 Web App

```bash
cd Day02
python app.py
# 浏览器打开 http://127.0.0.1:8089/
```

## 检索策略对比

| 策略 | 文件 | 思路 |
|------|------|------|
| 基础 | `rag.py` | query → embedding → top-k |
| Query 重写 | `rephraseRetrieve.py` | 结合历史重写后再检索 |
| 缓存 | `cacheRephraseRetrieve.py` | LRU + 语义命中,减少重复检索 |
| HyDE | `HyDERetrieve.py` | LLM 先生成假设答案,用答案做检索 |
| Multi-Query | `multiRetrieve.py` | LLM 生成多个子查询,union 结果 |
| Hybrid | `hybridRetrieve.py` | Dense + BM25(jieba 分词) |
| RRF | `RRFRetrieve.py` | 多路结果用倒数排名融合 |
| Model Rerank | `modelRRFRetrieve.py` | RRF 后再用 CrossEncoder 精排 |

## 评估

`Day02/metric.py` 基于 [RAGAS](https://github.com/explodinggradients/ragas) 输出:

- **Faithfulness** — 答案是否忠于检索上下文
- **Answer Relevancy** — 答案是否切题
- **Context Relevance / Groundedness** — 上下文相关性

## License

MIT
