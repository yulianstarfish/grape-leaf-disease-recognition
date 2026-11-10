# -*- coding: utf-8 -*-
"""实验2：LLM 回答质量三方案对比
A=纯LLM直接回答  B=基础RAG(512,k=5,无重排)  C=优化RAG(重排,配置取自实验1最优)
产出：实验结果/exp2_answers.xlsx（含自动评分+人工评分空白列，论文第6章人工盲评用）"""
import json
import time

import pandas as pd

import config
import llm_diagnosis
import rag_module

# 跑完实验1后，把最优配置填到这里（示例：512 分块 + 重排）
OPTIMAL = {"size": 512, "k": 5, "rerank": True}

JUDGE_PROMPT = (
    "你是评分员。根据【标准答案】给【系统回答】打分，五个维度各1-5分：\n"
    "准确性(与标准答案事实一致)、完整性(要点覆盖)、相关性(切题)、"
    "可操作性(建议具体可用)、一致性(无编造信息)。\n"
    "标准答案：{ref}\n系统回答：{ans}\n"
    '只输出JSON：{{"准确性": n, "完整性": n, "相关性": n, "可操作性": n, "一致性": n}}'
)


def auto_judge(ref: str, ans: str) -> dict:
    text = llm_diagnosis.chat([{"role": "user", "content": JUDGE_PROMPT.format(ref=ref, ans=ans)}],
                              temperature=0)
    try:
        return llm_diagnosis.parse_json(text)
    except Exception:
        return {}


def scheme_a(q):  # 纯 LLM
    return llm_diagnosis.chat([{"role": "user", "content": q}])


def scheme_b(q):  # 基础 RAG
    return rag_module.answer(q, size=512, k=5, rerank=False)["answer"]


def scheme_c(q):  # 优化 RAG
    return rag_module.answer(q, size=OPTIMAL["size"], k=OPTIMAL["k"],
                             rerank=OPTIMAL["rerank"])["answer"]


def main():
    import sys
    eval_set = json.load(open(config.BASE_DIR / "eval_set.json", encoding="utf-8"))
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else len(eval_set)
    eval_set = eval_set[:limit]
    print(f"评测集 {len(eval_set)} 条 × 3 方案 × 2 次调用（生成+评分）")

    rows = []
    for item in eval_set:
        for name, fn in [("A_纯LLM", scheme_a), ("B_基础RAG", scheme_b), ("C_优化RAG", scheme_c)]:
            ans = fn(item["question"])
            scores = auto_judge(item["reference_answer"], ans)
            row = {"问题id": item["id"], "类别": item["category"],
                   "问题": item["question"], "方案": name, "系统回答": ans}
            for dim in ["准确性", "完整性", "相关性", "可操作性", "一致性"]:
                row[f"自动_{dim}"] = scores.get(dim, "")
                row[f"人工_{dim}"] = ""   # 邀请盲评同学填写（1-5分）
            rows.append(row)
            print(f"q{item['id']} {name} 完成")
            time.sleep(0.3)

    pd.DataFrame(rows).to_excel(config.RESULT_DIR / "exp2_answers.xlsx",
                                index=False)  # 需 pip install openpyxl
    print(f"评分表已保存 → {config.RESULT_DIR / 'exp2_answers.xlsx'}")
    print("下一步：请3名同学对'系统回答'列盲评，填写人工_维度列，再算均值画柱状图")


if __name__ == "__main__":
    main()
