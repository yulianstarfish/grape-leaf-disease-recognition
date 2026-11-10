# -*- coding: utf-8 -*-
"""葡萄病害智能诊断系统 - Gradio Web 端（毕设系统主入口）
Tab1 图像诊断：上传叶片 → YOLOv8 识别 → LLM 生成诊断建议
Tab2 知识问答：RAG 检索知识库回答病虫害问题
运行：python app_gradio.py"""
import json

import gradio as gr
from ultralytics import YOLO

import config
import llm_diagnosis
import rag_module

model = YOLO(str(config.YOLO_MODEL))


def predict_and_diagnose(img, prompt_version):
    """图像 → 检测 → 取置信度最高的类别 → LLM 诊断"""
    if img is None:
        return "请先上传叶片图片", ""
    res = model.predict(img, verbose=False)[0]
    if res.probs is not None:
        # 分类模型：直接读取 top1 类别与置信度
        cls_id = int(res.probs.top1)
        conf = float(res.probs.top1conf)
    elif res.boxes is not None and len(res.boxes) > 0:
        # 检测模型：取置信度最高的检测框
        confs = res.boxes.conf.tolist()
        idx = confs.index(max(confs))
        cls_id = int(res.boxes.cls[idx])
        conf = confs[idx]
    else:
        return "未识别到叶片目标，请更换图片重试", ""
    disease = res.names[cls_id]

    try:
        diag = llm_diagnosis.diagnose(disease, conf, version=prompt_version)
        text = (json.dumps(diag, ensure_ascii=False, indent=2)
                if "text" not in diag else diag["text"])
    except Exception as e:  # API 未配置/网络异常时不影响识别结果展示
        text = f"【LLM 诊断调用失败：{e}\n请检查 config.py 中 LLM_API_KEY 是否已填写】"
    return f"识别结果：{disease}（置信度 {conf:.1%}）", text


def kb_qa(question):
    if not question.strip():
        return "请输入问题"
    try:
        out = rag_module.answer(question, size=512, k=5, rerank=False)
        return out["answer"]
    except Exception as e:
        return f"【问答失败：{e}\n请先运行 kb_builder.py 构建知识库】"


with gr.Blocks(title="葡萄叶片病害智能诊断系统") as demo:
    gr.Markdown("# 🍇 葡萄叶片病害智能诊断系统\nYOLOv8 图像识别 + 大语言模型（DeepSeek）诊断建议 + RAG 知识库问答")
    with gr.Tab("图像诊断"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="上传葡萄叶片照片")
                ver = gr.Radio(["v1", "v2", "v3"], value="v3",
                               label="Prompt 版本（论文消融实验用）")
                btn = gr.Button("开始诊断", variant="primary")
            with gr.Column():
                out_cls = gr.Textbox(label="识别结果")
                out_diag = gr.Textbox(label="诊断建议", lines=18)
        btn.click(predict_and_diagnose, [inp, ver], [out_cls, out_diag])
    with gr.Tab("知识库问答"):
        q = gr.Textbox(label="病虫害问题", placeholder="例：黑腐病用什么药防治？")
        btn2 = gr.Button("提问", variant="primary")
        a = gr.Textbox(label="回答", lines=12)
        btn2.click(kb_qa, q, a)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
