# RAG Course Assistant

一个面向课程学习场景的 Hybrid RAG 助教系统。项目提供混合检索、来源追踪、流式回答、多 Agent 路由、学习画像、评测与 benchmark，默认接入 AIHubMix 的 OpenAI 兼容接口。

## 功能

- Hybrid retrieval：Dense embedding + BM25 + RRF 融合召回
- Source-grounded QA：回答尽量引用文件名、页码和 chunk 编号
- Streaming API：`/chat_stream` 提供流式回答
- Multi-agent tutor：自动在问答、出题、错题分析、学习画像之间路由
- Evaluation：内置透明的 RAG 指标计算
- Benchmark：对比 Pure LLM 和 Source-grounded RAG
- Learning profile：根据最近提问生成学习画像和复习候选主题
- Document ingestion：支持 PDF、PPTX、DOCX、TXT，PDF/PPT 图片可选 OCR

## 默认模型配置

项目默认使用以下环境变量：

```env
OPENAI_API_BASE=https://aihubmix.com/v1
MODEL_NAME=alicloud-deepseek-v4-flash
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

如果你更想直接用另一条模型线，也可以把 `MODEL_NAME` 改成：

```env
MODEL_NAME=deepseek-v4-flash
```

## 安装

```bash
conda create -n rag-course-assistant python=3.10 -y
conda activate rag-course-assistant
pip install -r requirements.txt
```

然后创建 `.env`：

```env
OPENAI_API_KEY=your_aihubmix_api_key_here
OPENAI_API_BASE=https://aihubmix.com/v1
MODEL_NAME=alicloud-deepseek-v4-flash
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_TIMEOUT_SECONDS=60
```

不要把真实 API key 提交到仓库。

## 目录结构

```text
.
├── app.py
├── main.py
├── chat_store.py
├── backend/
│   ├── agents.py
│   ├── benchmark.py
│   ├── config.py
│   ├── document_loader.py
│   ├── learning_profile.py
│   ├── rag_agent.py
│   ├── rag_evaluator.py
│   ├── text_splitter.py
│   └── vector_store.py
├── frontend/
│   └── index.html
├── scripts/
│   ├── benchmark.py
│   ├── evaluate.py
│   └── process_data.py
└── datasets/
    └── eval_examples.jsonl
```

## 构建知识库

把课程资料放到 `data/` 目录，例如：

```text
data/
├── Lecture01.pdf
├── Slides02.pptx
└── notes.txt
```

然后运行：

```bash
python scripts/process_data.py
```

这一步会：

1. 加载文档
2. 做语义分块
3. 生成 embedding
4. 写入 `vector_db/`
5. 构建后续 Hybrid RAG 使用的 Chroma 数据

## 运行 Web 应用

```bash
uvicorn app:app --reload --port 8000
```

打开：

```text
http://127.0.0.1:8000
```

前端会展示：

- 当前模型与 embedding 配置
- API 联通状态
- 当前向量库文档数
- 本轮问题的检索来源
- 本地历史问题

## 主要接口

| Endpoint | Method | Description |
|---|---:|---|
| `/health` | GET | 返回当前服务、模型、embedding、向量库状态 |
| `/config` | GET | 返回运行时配置摘要 |
| `/chat_stream` | POST | 流式对话 |
| `/chat_with_sources` | POST | 返回回答和来源 |
| `/retrieve` | POST | 单独调试检索结果 |
| `/agent` | POST | 查看自动路由结果 |
| `/profile` | GET | 获取学习画像 |
| `/evaluate` | POST | 运行 RAG 评测 |
| `/benchmark` | POST | 对比 Pure LLM 与 RAG |
| `/analyze/recent` | POST | 基于最近问答生成学习总结 |
| `/analyze/mistakes` | POST | 基于最近问答生成错题分析 |

示例：

```bash
curl -X POST http://127.0.0.1:8000/chat_with_sources \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"请解释 Master Theorem 的三种情况\",\"top_k\":5}"
```

## 运行评测

```bash
python scripts/evaluate.py \
  --dataset datasets/eval_examples.jsonl \
  --output outputs/eval_results.json
```

输出指标包括：

- `answer_relevance`
- `context_precision`
- `citation_score`
- `f1`
- `final_score`

## 运行 Benchmark

```bash
python scripts/benchmark.py \
  --dataset datasets/eval_examples.jsonl \
  --output outputs/benchmark_results.json
```

该流程会对同一组问题分别跑：

1. Pure LLM
2. Source-grounded RAG

用于比较回答相关性、引用表现和上下文匹配程度。

## 备注

- 如果需要 OCR，请本地安装 Tesseract，并准备 `chi_sim`/`eng` 语言包。
- 如果 NLTK 缺少分词数据，可以运行 `python download_nltk_punkt.py`。
- 当前仓库没有自带课程资料，首次使用前需要自行准备 `data/` 目录内容。
