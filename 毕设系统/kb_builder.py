# -*- coding: utf-8 -*-
"""知识库构建：固定长度分块（256/512/1024 三种策略） + bge 中文 Embedding + Chroma 入库
论文第5章 RAG 构建 & 实验1 分块策略对比的基础"""
import config


def read_doc(path) -> str:
    """读取 md/txt；pdf 依赖 pypdf"""
    path = str(path)
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    return open(path, "r", encoding="utf-8").read()


def chunk(text: str, size: int = 512, overlap: int = 50) -> list:
    """固定长度滑动窗口分块，overlap 保留上下文衔接"""
    chunks = []
    for i in range(0, len(text), size - overlap):
        seg = text[i:i + size].strip()
        if len(seg) >= 30:  # 过滤过短的噪声片段
            chunks.append(seg)
    return chunks


def build(size: int, embed_model: str = None, embed_obj=None) -> "chromadb.Collection":
    """为指定分块长度构建向量库 collection（同名则复用）"""
    import chromadb
    from sentence_transformers import SentenceTransformer

    model_name = embed_model or config.EMBED_SMALL
    st = embed_obj or SentenceTransformer(model_name)

    docs = []
    for f in sorted(config.KB_DIR.glob("*")):
        if f.suffix.lower() in (".md", ".txt", ".pdf"):
            docs.extend(chunk(read_doc(f), size=size))
    print(f"[kb] 分块长度={size}, 语料块数={len(docs)}, embedding={model_name}")

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    col = client.get_or_create_collection(f"kb_{size}")
    vecs = st.encode(docs, normalize_embeddings=True).tolist()
    col.upsert(ids=[f"{size}_{i}" for i in range(len(docs))],
               documents=docs, embeddings=vecs)
    return col


if __name__ == "__main__":
    for s in config.CHUNK_SIZES:
        build(s)
    print("知识库构建完成 →", config.CHROMA_DIR)
