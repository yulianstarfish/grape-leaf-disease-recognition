"""
代码说明
pyqt5的颜色选择可使用pyqt5颜色选择.py来参考
主窗口的左上角x，左上角y，宽，高定义为win_x, win_y, win_w, win_h = 550, 400, 1500, 800
标题文本居中，y为30
图片展示区域image_x, image_y, image_w, image_h = 30, 100, 800, 500

巧思：
1.位置数据由推算得到，便于后续多次调整
2.按钮设置
3.动态折线图，完成后暂停
4.导出函数由弹窗命名（继承新的类出现弹窗）
"""
import os
import sys
import pandas as pd
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, \
    QTableWidget, QTableWidgetItem, QAbstractItemView, QDialog, QVBoxLayout, \
    QLineEdit, QDialogButtonBox, QMessageBox
from PyQt5.QtGui import QFont, QFontMetrics, QPixmap
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# 识别单张图片函数
def know_image(image_path):
    # 所需加载的模型目录（与脚本同目录的 best.pt）
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best.pt')
    # 加载模型
    model = YOLO(path, task='classify')
    # 检测图片，设置verbose为不输出日志
    results = model(image_path, verbose=False)
    # 获取分类概率
    probs = results[0].probs
    # 获取概率最高的分类的索引
    top1_index = probs.top1
    # 获取概率最高的分类的名称
    class_name = model.names[top1_index]
    # 获取概率最高的分类的概率值
    probability = probs.top1conf
    # 转为中文字典
    chinese = {"Grape___Black_rot": "黑腐病", "Grape___Esca_(Black_Measles)": "黑麻疹",
               "Grape___healthy": "健康", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "叶枯病"}
    return [chinese[class_name], f"{probability.item():.2%}", f"{sum(results[0].speed.values()):.2}ms"]


# 批量识别文件夹函数
def know_images(images_path):
    return [[os.path.join(images_path, img)] + know_image(os.path.join(images_path, img))
            for img in os.listdir(images_path)]


# 动态折线图函数
def pic():
    # 初始化函数，用于设置背景
    def init():
        line_pre.set_data([], [])
        return line_pre,

    # 更新函数，每次调用时更新数据
    def update(frame):
        # 添加新的 x 和 y 数据
        x_value = frame / 10
        # 添加x数据
        x_data.append(x_value * 10)
        # 添加y数据
        y_pre.append(df_y_pre[int(x_value * 10)])
        # 更新数据
        line_pre.set_data(x_data, y_pre)
        # 当更新到第149帧时，暂停动画
        if frame >= 149:
            ani.event_source.stop()
        return line_pre,

    # 读取数据（与脚本同目录的 results.csv，不存在则提示）
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')
    if not os.path.exists(csv_path):
        QMessageBox.warning(None, "提示", "未找到 results.csv，请重新训练模型后再查看模型效果")
        return
    df = pd.read_csv(csv_path)
    df_y_pre = df['metrics/accuracy_top1'].values
    # 数据初始化
    x_data = []
    y_pre = []

    # 设置字体为支持中文的字体
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    # 确保坐标轴刻度标签也使用该字体
    plt.rcParams['axes.unicode_minus'] = False
    # 创建图形和坐标轴，设置窗口大小为 10x6 英寸
    fig, ax = plt.subplots(figsize=(15, 10), dpi=100)
    fig.canvas.manager.window.setWindowTitle("模型效果展示窗口")
    line_pre, = ax.plot([], [], color='#2e2eff', label='每轮准确率', linewidth=5)
    # 标题
    plt.title("模型效果展示", fontdict={'size': 30})
    # 设置坐标轴范围
    ax.set_xlim(0, 160)
    # 增加一些空间以适应余弦线
    ax.set_ylim(0.98, 1.005)
    # x，y轴坐标大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # 添加图例
    ax.legend(fontsize=25)

    # 创建动画
    ani = FuncAnimation(fig, update, save_count=150, init_func=init, blit=True, interval=15)
    plt.xlabel('训练轮数', fontdict={'size': 25})
    plt.ylabel('准确率', fontdict={'size': 25})
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # 显示图形
    plt.show()

    # 保存
    # ani.save("animation.gif", fps=25, writer="imagemagick")


# 保存数据弹窗界面
class SaveWin(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        # 设置对话框的标题
        self.setWindowTitle("请输入文件名称")
        # 创建一个垂直布局
        layout = QVBoxLayout()

        # 创建一个单行文本输入框
        self.lineEdit = QLineEdit(self)
        # 将文本输入框添加到布局中
        layout.addWidget(self.lineEdit)
        # 创建一个按钮盒子，包含OK和Cancel按钮
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        # 设置OK按钮的文本为"确定并导出"
        box.button(QDialogButtonBox.Ok).setText("确定并导出")
        # 设置Cancel按钮的文本为"取消"
        box.button(QDialogButtonBox.Cancel).setText("取消")

        # 将按钮盒子添加到布局中
        layout.addWidget(box)
        # 将布局设置到对话框中
        self.setLayout(layout)
        self.setStyleSheet("""
            QDialog {
                background-color: #f0f0f0; /* 设置背景颜色为浅灰色 */
            }
            QLineEdit {
                background-color: #ffffff; /* 设置输入框背景颜色为白色 */
            }
            QPushButton {
                background-color: #c0c0c0; /* 设置按钮背景颜色为灰色 */
            }
        """)
        # 设置窗口的固定大小
        self.resize(400, 100)
        # 当点击确定并导出按钮时，连接到对话框的accept方法
        box.accepted.connect(self.accept)
        # 当点击取消按钮时，连接到对话框的reject方法
        box.rejected.connect(self.reject)


# pypt5主界面
class KnowLeaf(QWidget):
    def __init__(self):
        super().__init__()
        # 命名主要界面为init_ui
        self.init_ui()
        # 消除定义问题警告
        self.title_label = None
        self.image_b = None
        self.result_label = None
        # 表格行数默认为0
        self.table_row_count = 0

    def init_ui(self):
        # 窗口左上角x，左上角y，宽，高
        win_x, win_y, win_w, win_h = 550, 400, 1300, 800
        # 图片展示左上角x，左上角y，宽，高
        image_x, image_y, image_w, image_h = 30, 100, 800, 450
        # 展示区域的间隔和窗口间隔
        show_d, win_d = 20, 30
        # 结果和建议比例
        r, a = 3, 2
        # 多份结果展示位置表头比例
        r1, r2, r3, r4 = 2, 1, 1, 1
        # 结果展示左上角x，左上角y，宽，高
        result_x, result_y, result_w, result_h = image_x + image_w + show_d, \
            image_y, win_w - image_x * 2 - image_w - show_d, int((image_h - show_d) / (r + a) * r)
        # 建议展示左上角x，左上角y，宽，高
        advice_x, advice_y, advice_w, advice_h = result_x, result_y + result_h + show_d, \
            result_w, int(result_h / r * a)
        # 多份结果展示左上角x，左上角y，宽，高
        many_x, many_y, many_w, many_h = image_x, image_y + image_h + show_d, \
            image_w, win_h - image_y - image_h - show_d - win_d
        # 按钮展示左上角x，左上角y，宽，高
        button_x, button_y, button_w, button_h = result_x, many_y, \
            result_w, many_h
        # 按钮间距离
        button_b = 10

        # 背景颜色
        win_background = "#a5aeff"
        # 展示背景颜色
        show_background = "white"

        # 设置窗口标题
        self.setWindowTitle("智能识别葡萄叶系统")
        # 设置窗口大小（左上角x，左上角y，宽，高）
        self.setGeometry(win_x, win_y, win_w, win_h)
        # 设置背景颜色为紫色
        self.setStyleSheet(f"background-color: {win_background};")

        # "智能识别葡萄叶系统"标题文本
        self.title_label = QLabel("智能识别葡萄叶系统", self)
        # 设置字体（字体类型，大小，加粗）
        title_font = QFont("Arial", 20, QFont.Bold)
        # 设置标题文本格式为title_font
        self.title_label.setFont(title_font)
        # 标题文本宽度
        title_w = QFontMetrics(title_font).width(self.title_label.text())
        # 设置标题文本位置为居中（x，y）
        self.title_label.move((win_w - title_w) // 2, win_d)

        # 图片展示位置
        self.image_label = QLabel(self)
        # 设置图片展示位置大小（左上角x，左上角y，宽，高）
        self.image_label.setGeometry(image_x, image_y, image_w, image_h)
        # 设置QLabel的边框和背景色为白色
        self.image_label.setStyleSheet(f"background-color: {show_background};")
        # 默认背景图片（不存在则不显示，避免路径写死后无法启动）
        bg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '葡萄.png')
        self.show_image(bg_path)

        # 识别结果位置
        self.result_label = QLabel(self)
        # 设置图片展示位置大小（左上角x，左上角y，宽，高）
        self.result_label.setGeometry(result_x, result_y, result_w, result_h)
        # 设置QLabel的边框和背景色为白色
        self.result_label.setStyleSheet(f"background-color: {win_background};")

        # 识别结果类别文本，默认为”暂无结果“
        self.class_label = QLabel("暂无结果", self)
        # 设置字体（字体类型，大小，加粗）
        class_font = QFont("Arial", 20, QFont.Bold)
        # 设置标题文本格式为class_font
        self.class_label.setFont(class_font)
        # 使用样式表设置字体颜色为红色
        self.class_label.setStyleSheet("color: red;")
        # 类别文本宽度
        class_w = QFontMetrics(class_font).width(self.class_label.text())
        # 类别文本高度
        class_h = QFontMetrics(class_font).height()
        # 设置类别文本位置为居中（x，y）
        self.class_label.move(result_x + (result_w - class_w) // 2,
                              result_y + (result_h - class_h) // 2)
        # 识别结果置信度文本，默认为”置信度“
        self.conf_label = QLabel("置信度:" + " " * 20, self)
        # 设置字体（字体类型，大小，加粗）
        conf_font = QFont("Arial", 13, QFont.Bold)
        # 设置标题文本格式为conf_font
        self.conf_label.setFont(conf_font)
        # 置信度文本高度
        conf_h = QFontMetrics(conf_font).height()
        # 设置置信度文本位置为居中（x，y）
        self.conf_label.move(result_x,
                             result_y + result_h - conf_h)
        # 识别结果用时文本，默认为”用时“
        self.t_label = QLabel("用时:" + " " * 20, self)
        # 设置字体（字体类型，大小，加粗）
        t_font = QFont("Arial", 13, QFont.Bold)
        # 设置标题文本格式为t_font
        self.t_label.setFont(t_font)
        # 用时文本高度
        t_h = QFontMetrics(t_font).height()
        # 设置用时文本位置为居中（x，y）
        self.t_label.move(result_x + (result_w // 2),
                          result_y + result_h - t_h)

        # 建议展示位置
        self.advice_label = QLabel("防治方法与建议：", self)
        # 设置图片展示位置大小（左上角x，左上角y，宽，高）
        self.advice_label.setGeometry(advice_x, advice_y, advice_w, advice_h)
        # 设置自动换行
        self.advice_label.setWordWrap(True)
        # 设置对齐方式
        self.advice_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        # 设置字体样式
        self.advice_label.setStyleSheet(f"""
            QLabel {{
                font-family: KaiTi;
                font-size: 25px;
                line-height: 1.5; /* 行间距 */
                background-color: {show_background};
            }}
        """)

        # 多份结果展示位置
        # 初始化表格，0行4列，行数将动态添加
        self.many_t = QTableWidget(0, 4, self)
        # 设置图片展示位置大小（左上角x，左上角y，宽，高）
        self.many_t.setGeometry(many_x, many_y, many_w, many_h)
        self.many_t.setHorizontalHeaderLabels(['路径', '结果', '置信度', "用时"])
        # 设置表头对齐方式需要通过QHeaderView
        header = self.many_t.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignCenter)
        self.many_t.setStyleSheet(f"background-color: {show_background};")
        # 设置列宽和行高
        self.many_t.setColumnWidth(0, many_w // (r1 + r2 + r3 + r4) * r1)
        self.many_t.setColumnWidth(1, many_w // (r1 + r2 + r3 + r4) * r2)
        self.many_t.setColumnWidth(2, many_w // (r1 + r2 + r3 + r4) * r3)
        self.many_t.setColumnWidth(3, many_w // (r1 + r2 + r3 + r4) * r4)
        self.many_t.verticalHeader().setDefaultSectionSize(25)  # 设置行高
        # 设置选择模式为整行选择
        self.many_t.setSelectionBehavior(QAbstractItemView.SelectRows)
        # 设置选择模式为单选
        # self.many_t.setSelectionMode(QAbstractItemView.SingleSelection)
        # 设置点击效果
        self.many_t.itemClicked.connect(self.many_c)

        # 按钮展示位置
        button_width = (button_w - (2 - 1) * button_b) // 2
        button_height = (button_h - (2 - 1) * button_b) // 2
        # 按钮名称
        button_n = "打开图片 打开文件夹 模型效果 导出数据".split()
        # 2*2的按钮摆放
        for i in range(2):
            for j in range(2):
                button = QPushButton(f'{button_n[i * 2 + j + 1 - 1]}', self)
                # 依据位置放置按钮
                button.setGeometry(int(button_x + j * (button_width + button_b)),
                                   int(button_y + i * (button_height + button_b)),
                                   button_width, button_height)
                button.setStyleSheet("background-color: lightblue; "
                                     "color: black; font-size: 23px;")
                # 不同的按钮返回的结果不同，button_n对应的数字为1, 2, 3, 4
                button.clicked.connect(self.on_button_clicked(i * 2 + j + 1))

    # 不同按钮对应的功能
    def on_button_clicked(self, button_number):
        def slot():
            # 打开图片按钮
            if button_number == 1:
                self.open_image()
            # 打开文件夹按钮
            elif button_number == 2:
                self.open_folder()
            # 模型效果按钮
            elif button_number == 3:
                pic()
            # 导出数据按钮
            else:
                self.get_data()
        return slot

    # 打开图片函数
    def open_image(self):
        # 打开文件对话框，让用户选择图片
        options = QFileDialog.Options()
        image_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image files (*.png *.jpg *.jpeg)",
                                                    options=options)
        if image_path:
            # 预测图片
            i = know_image(image_path)
            # 添加到多行展示位置
            self.add_table_row(image_path, i[0], i[1], i[2])
            # 用户选择了图片，更新QLabel的图片
            self.show_image(image_path)
            # 更改文字
            self.change(i)

    # 显示图片函数
    def show_image(self, file_path):
        # 根据文件路径设置QLabel的图片
        pixmap = QPixmap(file_path)
        # 保持图片比例
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
        # 图片居中展示
        self.image_label.setAlignment(Qt.AlignCenter)

    # 改变文本函数
    def change(self, change_text):
        change_t = {"叶枯病": "喷施适宜的植物生长调节剂和杀菌剂，及时修剪枝条，增施有机肥改良土壤，加强通风排水以降低病害发生概率。",
                    "黑腐病": "选用抗病品种，加强检疫，冬季清园，及时排水修剪，增施有机肥，控制氮肥，果实套袋，生长季节摘除病果病叶，"
                    "关键时期喷施波尔多液等药剂。",
                    "黑麻疹": "选用抗病品种，严格消毒苗木，彻底清园并烧毁病残体，喷施杀菌剂预防，保持果园通风透光。",
                    "健康": "注重合理施肥、适时修剪以保持通风透光，定期检查生长状况，及时处理异常，冬季清园以减少病原菌越冬。", }
        self.class_label.setText(change_text[0])
        self.conf_label.setText("置信度:" + change_text[1])
        self.t_label.setText("用时:" + change_text[2])
        self.advice_label.setText("防治方法与建议：" + change_t[change_text[0]])

    # 打开文件夹函数
    def open_folder(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "QFileDialog.getExistingDirectory()", "", options=options)
        if folder_path:
            for i in know_images(folder_path):
                self.add_table_row(i[0], i[1], i[2], i[3])

    # 多行显示函数
    def add_table_row(self, path, result, confidence, result_t):
        self.many_t.insertRow(self.table_row_count)
        self.many_t.setItem(self.table_row_count, 0, QTableWidgetItem(path))
        self.many_t.setItem(self.table_row_count, 1, QTableWidgetItem(result))
        self.many_t.setItem(self.table_row_count, 2, QTableWidgetItem(confidence))
        self.many_t.setItem(self.table_row_count, 3, QTableWidgetItem(result_t))
        self.table_row_count += 1

    # 点击效果
    def many_c(self, item):
        # 获取点击的单元格的内容
        text = [self.many_t.item(item.row(), column).text() for column in range(self.many_t.columnCount())]
        # 设置图片
        self.show_image(text[0])
        # 结果
        self.change(text[1:])
        # print(text)

    # 导出数据函数函数
    def get_data(self):
        self.win_save = SaveWin(self)
        if self.win_save.exec_() == QDialog.Accepted:
            user_input = self.win_save.lineEdit.text()
            # 获取数据并保存
            # 获取列标题
            columns = [self.many_t.horizontalHeaderItem(i).text() for i in range(self.many_t.columnCount())]
            # 获取行数据
            rows = []
            for row_index in range(self.many_t.rowCount()):
                row_data = []
                for column_index in range(self.many_t.columnCount()):
                    item = self.many_t.item(row_index, column_index)
                    if item is not None:
                        row_data.append(item.text())
                    else:
                        row_data.append(None)
                rows.append(row_data)
            # 创建DataFrame
            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(user_input + ".csv", index=False)


def main():
    app = QApplication(sys.argv)
    ex = KnowLeaf()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
