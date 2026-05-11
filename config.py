import torch
import os

# ===================== 全局基础配置 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# HuggingFace 镜像与缓存配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(BASE_DIR, "huggingface_cache")

# 文档处理配置 
DOCS_INPUT_DIR = os.path.join(BASE_DIR, "data/doc_raw")
CHUNKS_OUTPUT_PATH = os.path.join(BASE_DIR, "data/result/doc_chunks.pkl")
MIN_PARAGRAPH_LENGTH = 20

#  检索引擎配置 
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
BM25_CHUNKS_PATH = os.path.join(BASE_DIR, "data/chunks_for_bm25.pkl")
TOP_K_RETRIEVE = 5  # 初始检索数量
MAX_CONTEXT_NUM = 4  # 最终保留的上下文数量

# 大模型配置 
LLM_MODEL_PATH = os.path.join(BASE_DIR, "model/chatglm2-6b")
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
TOP_P = 0.75
REPETITION_PENALTY = 1.2
NO_REPEAT_NGRAM_SIZE = 3
LENGTH_PENALTY = 1.0

#  Gradio 界面配置 
GRADIO_SERVER_PORT = 7860
GRADIO_SHARE = False
