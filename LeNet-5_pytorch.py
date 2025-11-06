# -*- coding: utf-8 -*-
"""
MNIST 卷积核可视化 GUI（单窗口集成版）
---------------------------------------------------
左侧：卷积核对比图（嵌入 matplotlib）
右侧：按钮 + 日志（训练控制区）
"""

import os, random, threading, torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import tkinter as tk
from tkinter import scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# ====== 字体支持 ======
if os.name == "nt":
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
else:
    plt.rcParams["font.sans-serif"] = ["PingFang TC", "Noto Sans CJK"]
plt.rcParams["axes.unicode_minus"] = False


# ====================
# 简易 CNN 模型
# ====================
class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 3, 1)
        self.fc1 = nn.Linear(26 * 26 * 5, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc1(x)


# ====================
# 主应用类
# ====================
class ConvCompareApp:
    def __init__(self, master):
        self.master = master
        self.master.title("MNIST 卷积核前后对比（单窗口布局）")
        self.master.geometry("1100x650")

        # 数据
        transform = transforms.Compose([transforms.ToTensor()])
        self.trainset = datasets.MNIST("./MNIST_data", train=True, download=True, transform=transform)
        self.testset = datasets.MNIST("./MNIST_data", train=False, download=True, transform=transform)
        self.idx = 0
        self.image, self.label = self.trainset[self.idx]

        # 模型
        self.model = MiniCNN()
        self.device = torch.device("cpu")
        self.initial_kernels = self.model.conv1.weight.detach().clone()
        self.epoch_counter = 1

        # 创建主布局（左：图像；右：控制区）
        self.create_layout()

        # 初次绘图
        self.update_plot()

    # ====================
    # 布局设置
    # ====================
    def create_layout(self):
        # 主容器
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧 matplotlib 图像
        self.fig, self.axes = plt.subplots(5, 4, figsize=(8, 8))
        self.fig.suptitle("初始与训练后卷积核对比", fontsize=14)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        # 右侧按钮 + 日志
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.grid(row=0, column=1, sticky="ns", padx=10, pady=10)

        # 按钮
        self.train_btn = tk.Button(self.control_frame, text=f"训练模型 ({self.epoch_counter} epoch)",
                                   command=self.start_training_thread, width=18)
        self.train_btn.pack(pady=5)

        tk.Button(self.control_frame, text="上一张", command=self.prev_img, width=18).pack(pady=3)
        tk.Button(self.control_frame, text="下一张", command=self.next_img, width=18).pack(pady=3)
        tk.Button(self.control_frame, text="随机一张", command=self.random_img, width=18).pack(pady=3)
        tk.Button(self.control_frame, text="退出", command=self.master.quit, width=18).pack(pady=3)

        # 滚动日志区
        self.log_box = scrolledtext.ScrolledText(self.control_frame, width=40, height=25, font=("Consolas", 9))
        self.log_box.pack(pady=10, fill=tk.BOTH, expand=True)
        self.log("等待训练...")

        # 调整主区域比例
        self.main_frame.columnconfigure(0, weight=3)
        self.main_frame.columnconfigure(1, weight=1)

    # ====================
    # 日志显示
    # ====================
    def log(self, text):
        self.log_box.insert(tk.END, text + "\n")
        self.log_box.see(tk.END)
        self.master.update_idletasks()

    # ====================
    # 多线程训练
    # ====================
    def start_training_thread(self):
        t = threading.Thread(target=self.train_one_epoch)
        t.daemon = True
        t.start()

    def train_one_epoch(self):
        self.train_btn.config(state=tk.DISABLED)
        self.log(f"⏳ 正在训练第 {self.epoch_counter} 个 epoch...")

        train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.testset, batch_size=1000, shuffle=False)
        opt = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            opt.zero_grad()
            out = self.model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

            if (i + 1) % 200 == 0:
                avg_loss = total_loss / (i + 1)
                msg = f"Step {i+1}, avg_loss={avg_loss:.4f}"
                print(msg)
                self.log(msg)

        # 测试准确率
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total * 100
        self.log(f"✅ 第 {self.epoch_counter} 个 epoch 训练完成，测试集准确率：{acc:.2f}%\n")

        self.epoch_counter += 1
        self.master.after(0, lambda: self.train_btn.config(
            state=tk.NORMAL, text=f"训练模型 ({self.epoch_counter} epoch)"
        ))
        self.master.after(0, self.update_plot)

    # ====================
    # 图片控制
    # ====================
    def prev_img(self):
        self.idx = (self.idx - 1) % len(self.trainset)
        self.image, self.label = self.trainset[self.idx]
        self.update_plot()

    def next_img(self):
        self.idx = (self.idx + 1) % len(self.trainset)
        self.image, self.label = self.trainset[self.idx]
        self.update_plot()

    def random_img(self):
        self.idx = random.randint(0, len(self.trainset) - 1)
        self.image, self.label = self.trainset[self.idx]
        self.update_plot()

    # ====================
    # 绘图逻辑
    # ====================
    def update_plot(self):
        img = self.image.numpy()[0]
        init_kernels = self.initial_kernels.cpu().numpy()
        trained_kernels = self.model.conv1.weight.detach().cpu().numpy()

        for ax_row in self.axes:
            for ax in ax_row:
                ax.clear()

        for i in range(5):
            # 初始卷积核
            self._draw_kernel(self.axes[i, 0], init_kernels[i, 0], f"初始核 {i+1}")
            # 初始结果
            y1 = self.apply_conv(img, init_kernels[i, 0])
            self.axes[i, 1].imshow(y1, cmap="gray")
            self.axes[i, 1].axis("off")
            self.axes[i, 1].set_title(f"初始结果 {i+1}")

            # 训练后卷积核
            self._draw_kernel(self.axes[i, 2], trained_kernels[i, 0], f"训练核 {i+1}")
            # 训练后结果
            y2 = self.apply_conv(img, trained_kernels[i, 0])
            self.axes[i, 3].imshow(y2, cmap="gray")
            self.axes[i, 3].axis("off")
            self.axes[i, 3].set_title(f"训练结果 {i+1}")

        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05, wspace=0.3, hspace=0.3)
        self.canvas.draw_idle()

    def _draw_kernel(self, ax, kernel, title):
        ax.imshow(kernel, cmap="gray", interpolation="nearest", vmin=-0.5, vmax=0.5)
        ax.axis("off")
        ax.set_title(title)
        for r in range(kernel.shape[0]):
            for c in range(kernel.shape[1]):
                val = kernel[r, c]
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        color="red" if abs(val) > 0.3 else "black", fontsize=8, weight="bold")

    def apply_conv(self, img, kernel):
        k = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        y = F.conv2d(x, k, padding=1)
        y = y.squeeze().detach().numpy()
        y = (y - y.min()) / (y.max() - y.min() + 1e-8)
        return y


# ====================
# 主入口
# ====================
if __name__ == "__main__":
    root = tk.Tk()
    app = ConvCompareApp(root)
    root.mainloop()
