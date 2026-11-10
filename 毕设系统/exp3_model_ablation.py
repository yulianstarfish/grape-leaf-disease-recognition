# -*- coding: utf-8 -*-
"""实验3：识别模块模型消融对比（YOLOv8n vs YOLOv8s，或增强开/关两组训练结果）
产出：实验结果/exp3_results.csv + 指标对比图 + 混淆矩阵（论文第4/6章）
使用前在 config.py 填好 DATA_YAML；训练 YOLOv8s：
    yolo train model=yolov8s.pt data=<你的yaml> epochs=100 imgsz=640
"""
import pandas as pd

import config


def evaluate(weights: str, name: str):
    from ultralytics import YOLO
    model = YOLO(weights)
    metrics = model.val(data=config.DATA_YAML, plots=True)  # plots 生成混淆矩阵/PR曲线
    speed = metrics.speed  # ms/张
    fps = 1000 / (speed.get("inference", 1) or 1)
    row = {
        "模型": name,
        "mAP@0.5": round(float(metrics.box.map50), 4),
        "mAP@0.5:0.95": round(float(metrics.box.map), 4),
        "Precision": round(float(metrics.box.mp), 4),
        "Recall": round(float(metrics.box.mr), 4),
        "推理耗时ms/张": round(speed.get("inference", 0), 2),
        "FPS": round(fps, 1),
        "参数量M": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2),
    }
    print(row)
    return row


def main():
    if not config.DATA_YAML:
        raise SystemExit("请先在 config.py 中填写 DATA_YAML（数据集 yaml 路径）")

    tasks = [(str(config.YOLO_MODEL), "YOLOv8n")]
    if config.YOLO_MODEL_S:
        tasks.append((config.YOLO_MODEL_S, "YOLOv8s"))
    else:
        print("提示：未配置 YOLO_MODEL_S，仅评测 YOLOv8n；训练完成后填入再跑一次即可对比")

    rows = [evaluate(w, n) for w, n in tasks]
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULT_DIR / "exp3_results.csv", index=False, encoding="utf-8-sig")

    if len(rows) > 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        metrics = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"]
        fig, axes = plt.subplots(1, len(metrics) + 1, figsize=(4 * len(metrics) + 4, 3.8))
        for ax, m in zip(axes, metrics):
            ax.bar(df["模型"], df[m], color=["#4c72b0", "#dd8452"])
            ax.set_title(m); ax.set_ylim(0, 1); ax.grid(alpha=.3, axis="y")
        axes[-1].bar(df["模型"], df["FPS"], color=["#4c72b0", "#dd8452"])
        axes[-1].set_title("FPS"); axes[-1].grid(alpha=.3, axis="y")
        fig.tight_layout()
        fig.savefig(config.RESULT_DIR / "exp3_compare.png", dpi=200)
    print(f"结果已保存 → {config.RESULT_DIR}")


if __name__ == "__main__":
    main()
