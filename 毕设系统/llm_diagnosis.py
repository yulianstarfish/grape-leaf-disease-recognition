# -*- coding: utf-8 -*-
"""LLM 智能诊断模块：DeepSeek API + Prompt 工程（v1/v2/v3 三版迭代，供论文第5章与消融展示）"""
import json
import re
from openai import OpenAI
import config


def _client():
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)


# ============ Prompt 迭代记录（答辩亮点：展示优化过程） ============
PROMPTS = {
    # v1：最朴素的直接提问（基线）
    "v1": (
        "葡萄叶片检测到病害：{disease}，置信度{conf}。"
        "请给出病害解释和防治建议。"
    ),
    # v2：要求结构化 JSON 输出，便于程序解析展示
    "v2": (
        "你是葡萄病虫害防治助手。葡萄叶片检测到病害：{disease}，置信度{conf}。\n"
        "请严格按照如下 JSON 格式输出，不要输出其他内容：\n"
        '{{"病害名称": "", "病害解释": "", "严重程度": "轻/中/重", "防治建议": ["", ""]}}'
    ),
    # v3：角色设定 + 领域约束 + 输出规范（最终版）
    "v3": (
        "你是一名经验丰富的葡萄种植与病虫害防治专家，为果农提供专业、可操作的诊断建议。\n"
        "图像识别结果：病害类别「{disease}」，模型置信度{conf}。\n"
        "要求：\n"
        "1. 仅针对该病害作答，不要诊断其他病害；\n"
        "2. 解释需包含病原类型、典型症状、发病规律；\n"
        "3. 建议按 农业防治/化学防治 分列，具体到药剂名称与使用频次，并提醒安全间隔期；\n"
        "4. 置信度低于70%时需提示用户结合人工复核；\n"
        "5. 严格输出如下 JSON，不要输出任何其他文字：\n"
        '{{"病害名称": "", "病原类型": "", "症状描述": "", "发病规律": "", '
        '"严重程度": "轻/中/重", "农业防治": ["", ""], "化学防治": ["", ""], "注意事项": ""}}'
    ),
}


def parse_json(text: str) -> dict:
    """从模型输出中提取第一个 JSON 对象并解析，失败时返回原文兜底"""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"解析失败_原文": text}


def diagnose(disease: str, conf: float, version: str = "v3") -> dict:
    """图像识别结果 → LLM 诊断建议。version 用于论文 Prompt 消融对比"""
    prompt = PROMPTS[version].format(disease=disease, conf=f"{conf:.1%}")
    resp = _client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    text = resp.choices[0].message.content
    return {"text": text} if version == "v1" else parse_json(text)


def chat(messages: list, temperature: float = 0.3) -> str:
    """通用多轮对话接口（RAG 问答与实验脚本共用）"""
    resp = _client().chat.completions.create(
        model=config.LLM_MODEL, messages=messages, temperature=temperature
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    result = diagnose("Black_rot黑腐病", 0.95, version="v3")
    print(json.dumps(result, ensure_ascii=False, indent=2))
