# 中文法规问答 RAG 系统

基于 LangChain 构建的检索增强生成（RAG）问答系统，面向中文法规语料，提供从文档入库、多策略检索到流式问答 Web App 的完整链路，并通过 RAGAS 进行质量评估。

- **Embedding**：`bge-base-zh-v1.5`（本地）
- **LLM**：阿里通义千问 `qwen-turbo`（DashScope）
- **Vector Store**：Chroma
- **语料**：`knowledge_base/` 下的 txt / md / pdf / docx 文档

## 目录结构

```
.
├── notebooks/                      # 探索性 notebooks
│   ├── document_loading.ipynb      # TextLoader 加载单个 txt
│   ├── multi_format_loading.ipynb  # 多格式加载器 (txt/md/pdf/docx)
│   ├── vector_retrieval.ipynb      # bge embedding + Chroma 相似度检索
│   ├── lcel_basics.ipynb           # LCEL: prompt | LLM | parser
│   ├── lcel_history.ipynb          # LCEL + 对话历史
│   └── lcel_advanced.ipynb         # LCEL 进阶组合
│
├── src/                            # 应用主代码
│   ├── indexing.py                 # 文档入库: 加载/清洗/切分(400/40)/持久化
│   ├── rag.py                      # 异步流式 RAG 入口
│   ├── rephraseRetrieve.py         # 基于对话历史的 query 重写
│   ├── cacheRephraseRetrieve.py    # LRU + 语义缓存 (FAISS/SHA256)
│   ├── HyDERetrieve.py             # HyDE: 生成假设答案再检索
│   ├── multiRetrieve.py            # 多查询检索 (LLM 生成子查询)
│   ├── hybridRetrieve.py           # 混合检索: Dense (Chroma) + Sparse (BM25/jieba)
│   ├── RRFRetrieve.py              # Reciprocal Rank Fusion
│   ├── modelRRFRetrieve.py         # RRF + CrossEncoder (bge-reranker-base) 重排
│   ├── metric.py                   # RAGAS 评估 (faithfulness / relevancy / ...)
│   ├── app.py                      # FastAPI + SSE 流式问答 Web App (127.0.0.1:8089)
│   └── templates/
│       └── naive_index.html        # 前端聊天 UI
│
├── knowledge_base/                 # 语料库
│   ├── sample.txt
│   ├── sample.md
│   ├── sample.pdf
│   └── sample.docx
│
├── bge-base-zh-v1.5/               # 本地 embedding 模型 (gitignored)
└── bge-reranker-base/              # 本地 reranker 模型 (gitignored)
```

## 快速开始

### 1. 环境要求

- Python 3.10+（代码用到 `list[Document]` 等 PEP 585 语法）
- 依赖（无 requirements.txt，按需安装）：

```bash
pip install langchain langchain-chroma langchain-huggingface langchain-community \
            chromadb faiss-cpu jieba torch transformers datasets ragas \
            unstructured fastapi uvicorn python-dotenv
```

### 2. 下载本地模型

仓库未包含模型权重（GitHub 单文件 100MB 限制），请从 HuggingFace 拉取放到项目根目录：

```bash
# embedding 模型
git clone https://huggingface.co/BAAI/bge-base-zh-v1.5

# reranker 模型（仅 modelRRFRetrieve.py 需要）
git clone https://huggingface.co/BAAI/bge-reranker-base
```

或使用 modelscope 镜像。

### 3. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 DashScope API Key
```

`.env` 示例：

```
TONGYI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> 申请地址：https://dashscope.console.aliyun.com/

### 4. 构建索引

```bash
cd src
python indexing.py
```

会读取 `knowledge_base/` 下的文档，切分后写入 `src/vectorstore/`。

### 5. 启动 Web App

```bash
cd src
python app.py
# 浏览器打开 http://127.0.0.1:8089/
```

## 检索策略

| 策略 | 文件 | 思路 |
|------|------|------|
| 基础 | `rag.py` | query → embedding → top-k |
| Query 重写 | `rephraseRetrieve.py` | 结合历史重写后再检索 |
| 缓存 | `cacheRephraseRetrieve.py` | LRU + 语义命中，减少重复检索 |
| HyDE | `HyDERetrieve.py` | LLM 先生成假设答案，用答案做检索 |
| Multi-Query | `multiRetrieve.py` | LLM 生成多个子查询，union 结果 |
| Hybrid | `hybridRetrieve.py` | Dense + BM25（jieba 分词） |
| RRF | `RRFRetrieve.py` | 多路结果用倒数排名融合 |
| Model Rerank | `modelRRFRetrieve.py` | RRF 后再用 CrossEncoder 精排 |

## 评估

`src/metric.py` 基于 [RAGAS](https://github.com/explodinggradients/ragas) 输出：

- **Faithfulness** — 答案是否忠于检索上下文
- **Answer Relevancy** — 答案是否切题
- **Context Relevance / Groundedness** — 上下文相关性

## License

MIT