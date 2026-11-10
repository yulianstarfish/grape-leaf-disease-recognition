# -*- coding: utf-8 -*-
"""实验1：RAG 检索优化对比
对比维度：分块长度(256/512/1024) × Top-K(3/5/8) × 是否重排
指标：检索命中率@k（LLM 判定检索片段能否支撑回答该问题）
产出：实验结果/exp1_results.csv + 命中率对比图（论文第6章）
运行前提：已配置 LLM_API_KEY；首次运行自动下载 bge 模型并构建向量库"""
import json
import time
from itertools import product

import pandas as pd

import config
import kb_builder
import llm_diagnosis
import rag_module

HIT_PROMPT = (
    "判断以下检索片段是否包含回答【问题】所需的关键信息。\n"
    "问题：{q}\n标准答案要点：{ref}\n检索片段：{docs}\n"
    "只回答 是 或 否。"
)


def judge_hit(question: str, ref: str, docs: list) -> int:
    """LLM-as-Judge：检索片段是否可支撑回答（1/0）"""
    text = llm_diagnosis.chat([{"role": "user", "content": HIT_PROMPT.format(
        q=question, ref=ref, docs="\n".join(f"[{i+1}]{d[:300]}" for i, d in enumerate(docs)))}],
        temperature=0)
    return 1 if "是" in text[:5] else 0


def main():
    eval_set = json.load(open(config.BASE_DIR / "eval_set.json", encoding="utf-8"))
    # 支持用 --limit N 快速试跑，跑全量请用全部条目
    import sys
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else len(eval_set)
    eval_set = eval_set[:limit]
    print(f"评测集：{len(eval_set)} 条；完整运行约 {len(eval_set)*18} 次 LLM 判定，注意 API 花费")

    rows = []
    embed = rag_module._embed(config.EMBED_SMALL)  # 提前加载
    for size in config.CHUNK_SIZES:
        col = kb_builder.build(size, embed_obj=embed)
        for rerank in config.USE_RERANK:
            for item in eval_set:
                docs8 = rag_module.retrieve(
                    item["question"], size=size, k=8, rerank=rerank)
                for k in config.TOP_KS:
                    hit = judge_hit(item["question"], item["reference_answer"], docs8[:k])
                    rows.append({"分块长度": size, "重排": rerank, "TopK": k,
                                 "问题id": item["id"], "命中": hit})
                    print(f"size={size} rerank={rerank} k={k} q{item['id']} hit={hit}")
                    time.sleep(0.3)  # 控制请求频率

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULT_DIR / "exp1_results.csv", index=False, encoding="utf-8-sig")

    summary = df.groupby(["分块长度", "重排", "TopK"])["命中"].mean().reset_index()
    summary.columns = ["分块长度", "是否重排", "Top-K", "检索命中率"]
    summary.to_csv(config.RESULT_DIR / "exp1_summary.csv", index=False, encoding="utf-8-sig")
    print("\n==== 检索命中率汇总 ====")
    print(summary.to_string(index=False))

    # 出图：不同分块长度下 命中率-TopK 曲线
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, rr in zip(axes, config.USE_RERANK):
        for size in config.CHUNK_SIZES:
            sub = summary[(summary["分块长度"] == size) & (summary["是否重排"] == rr)]
            ax.plot(sub["Top-K"], sub["检索命中率"], marker="o", label=f"分块{size}")
        ax.set_title("bge-reranker 重排" if rr else "无重排")
        ax.set_xlabel("Top-K"); ax.grid(alpha=.3); ax.legend()
    axes[0].set_ylabel("检索命中率")
    fig.tight_layout()
    fig.savefig(config.RESULT_DIR / "exp1_hit_rate.png", dpi=200)
    print(f"结果已保存 → {config.RESULT_DIR}")


if __name__ == "__main__":
    main()
