# -*- coding: utf-8 -*-
"""RAG 检索问答模块：向量检索 + 可选 bge 重排 + LLM 生成带引用来源的回答"""
import config

_ST = {}          # embedding 模型缓存
_RR = None        # 重排模型缓存


def _embed(model_name: str):
    if model_name not in _ST:
        from sentence_transformers import SentenceTransformer
        _ST[model_name] = SentenceTransformer(model_name)
    return _ST[model_name]


def _reranker():
    global _RR
    if _RR is None:
        from FlagEmbedding import FlagReranker
        _RR = FlagReranker(config.RERANKER, use_fp16=True)
    return _RR


def get_collection(size: int):
    import chromadb
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(f"kb_{size}")


def retrieve(query: str, size: int = 512, k: int = 5,
             embed_model: str = None, rerank: bool = False) -> list:
    """检索最相关的 k 个知识片段；rerank=True 时用 bge-reranker 精排"""
    st = _embed(embed_model or config.EMBED_SMALL)
    col = get_collection(size)
    res = col.query(query_embeddings=[st.encode([query], normalize_embeddings=True)[0].tolist()],
                    n_results=max(k, 20 if rerank else k))
    docs = res["documents"][0]
    if rerank and docs:
        scores = _reranker().compute_score([[query, d] for d in docs], normalize_scores=True)
        docs = [d for _, d in sorted(zip(scores, docs), key=lambda x: -x[0])]
    return docs[:k]


def answer(question: str, size: int = 512, k: int = 5,
           embed_model: str = None, rerank: bool = False) -> dict:
    """RAG 回答：检索 → 拼 Prompt（要求基于资料+标注依据）→ LLM 生成"""
    from llm_diagnosis import chat
    docs = retrieve(question, size, k, embed_model, rerank)
    ctx = "\n".join(f"[资料{i+1}] {d}" for i, d in enumerate(docs))
    prompt = (
        "你是葡萄病虫害防治专家。请仅依据以下知识库资料回答果农问题，"
        "答案末尾用（依据：资料N）标注所引资料编号；资料不足以回答时明确说明。\n\n"
        f"{ctx}\n\n问题：{question}"
    )
    ans = chat([{"role": "user", "content": prompt}])
    return {"answer": ans, "sources": docs[:k]}


if __name__ == "__main__":
    out = answer("黑腐病用什么药防治？", k=5)
    print(out["answer"])
