# -*- coding: utf-8 -*-
"""毕设系统全局配置：路径、API 密钥、实验参数统一管理"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent                      # 毕设系统目录
PROJECT_DIR = BASE_DIR.parent                         # 葡萄叶片病害项目根目录
RESULT_DIR = BASE_DIR / "实验结果"                     # 实验输出（表格/图表）
RESULT_DIR.mkdir(exist_ok=True)

# 从当前目录的 .env 加载密钥；.env 不进 Git，防止泄露
load_dotenv(BASE_DIR / ".env")

# ================= 模型 =================
YOLO_MODEL = PROJECT_DIR / "best.pt"    # 已有 YOLOv8n 权重（识别模块用）
YOLO_MODEL_S = r""                      # 实验3：若训练了 YOLOv8s，填其 best.pt 路径

# 实验3 评测用数据集 yaml（ultralytics 格式，含 train/val 路径与 4 类别）
DATA_YAML = r""                         # 例: r"D:\个人\项目\数据集\grape.yaml"

# ================= LLM（DeepSeek）=================
# 注册地址 https://platform.deepseek.com ，创建 API Key 后写入本目录 .env：
# DEEPSEEK_API_KEY=sk-你的key
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"

# ================= 知识库（RAG）=================
KB_DIR = BASE_DIR / "知识库"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Embedding 模型（实验1对比项之一；bge-large 体积较大，可选）
EMBED_SMALL = "BAAI/bge-small-zh-v1.5"
EMBED_LARGE = "BAAI/bge-large-zh-v1.5"
RERANKER = "BAAI/bge-reranker-base"

# 实验1默认对比维度
CHUNK_SIZES = [256, 512, 1024]
TOP_KS = [3, 5, 8]
USE_RERANK = [False, True]
