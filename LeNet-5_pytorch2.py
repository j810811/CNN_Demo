# -*- coding: utf-8 -*-
"""
LeNet-5 卷积核可视化 GUI（横向滚动版）
---------------------------------------------------------
特征：
- 单窗口 + C3/S4 双向滚动
- 去除多余边距，左右画布更贴近
- 保持所有功能完整（训练、切换、日志）
"""

import os, random, threading
import tkinter as tk
from tkinter import scrolledtext, ttk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# === 字体（支持中文） ===
if os.name == "nt":
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
else:
    plt.rcParams["font.sans-serif"] = ["PingFang TC", "Noto Sans CJK"]
plt.rcParams["axes.unicode_minus"] = False

# === LeNet-5（28×28 输入适配） ===
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 0)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# === 主应用 ===
class LeNet5VisualizerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("LeNet-5 卷积核可视化（横向滚动版）")
        self.master.geometry("1380x780")

        # 数据集
        tf = transforms.Compose([transforms.ToTensor()])
        self.trainset = datasets.MNIST("./MNIST_data", train=True, download=True, transform=tf)
        self.testset = datasets.MNIST("./MNIST_data", train=False, download=True, transform=tf)
        self.idx = 0
        self.image, self.label = self.trainset[self.idx]

        # 模型与初始权重
        self.model = LeNet5()
        self.device = torch.device("cpu")
        self.c1_init = self.model.conv1.weight.detach().clone()
        self.epoch_counter = 1

        self._build_layout()
        self._update_all_plots()

    # === 布局 ===
    def _build_layout(self):
        self.main = tk.Frame(self.master, bg="#ffffff")
        self.main.pack(fill=tk.BOTH, expand=True)

        self.left_frame = tk.Frame(self.main, bg="#ffffff")
        self.ctrl_frame = tk.Frame(self.main, bg="#ffffff")

        self.main.columnconfigure(0, weight=10)
        self.main.columnconfigure(1, weight=0)
        self.main.rowconfigure(0, weight=1)

        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.ctrl_frame.grid(row=0, column=1, sticky="ns", padx=(4, 4), pady=0)
        self.ctrl_frame.config(width=220)

        self.left_frame.columnconfigure(0, weight=1)
        self.left_frame.columnconfigure(1, weight=0)
        self.left_frame.rowconfigure(0, weight=1)

        # ---- 左画布 ----
        self.fig_left, self.axes_left = plt.subplots(6, 5, figsize=(8, 8))
        self.fig_left.suptitle("C1 初始/训练后 + 卷积结果 + S2 池化", fontsize=12)
        self.canvas_left = FigureCanvasTkAgg(self.fig_left, master=self.left_frame)
        self.canvas_left.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # ---- 右画布（双向滚动）----
        self.right_canvas = tk.Canvas(self.left_frame, bg="#ffffff", highlightthickness=0, bd=0)
        self.right_canvas.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)

        self.right_scrollbar_y = ttk.Scrollbar(self.left_frame, orient="vertical", command=self.right_canvas.yview)
        self.right_scrollbar_y.grid(row=0, column=2, sticky="ns", padx=0, pady=0)

        self.right_scrollbar_x = ttk.Scrollbar(self.left_frame, orient="horizontal", command=self.right_canvas.xview)
        self.right_scrollbar_x.grid(row=1, column=1, sticky="ew", padx=0, pady=(0, 0))

        self.right_canvas.configure(xscrollcommand=self.right_scrollbar_x.set,
                                    yscrollcommand=self.right_scrollbar_y.set)

        self.right_inner = tk.Frame(self.right_canvas, bg="#ffffff")
        self.right_canvas_window = self.right_canvas.create_window((0, 0), window=self.right_inner, anchor="nw")

        self.fig_right, self.axes_right = plt.subplots(16, 3, figsize=(2, 22))
        self.fig_right.suptitle("C3 核 / C3 卷积 / S4 池化", fontsize=12)
        self.canvas_right = FigureCanvasTkAgg(self.fig_right, master=self.right_inner)
        self.canvas_right.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        def _on_configure(event):
            self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))
        self.right_inner.bind("<Configure>", _on_configure)

        self.right_canvas.bind_all("<MouseWheel>",
                                   lambda e: self.right_canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        # ================= 控制区 =================
        self.train_btn = tk.Button(self.ctrl_frame,
                                   text=f"训练模型 ({self.epoch_counter} epoch)",
                                   command=self._start_training_thread, width=18)
        self.train_btn.pack(pady=2)
        tk.Button(self.ctrl_frame, text="上一张", command=self._prev_img, width=18).pack(pady=1)
        tk.Button(self.ctrl_frame, text="下一张", command=self._next_img, width=18).pack(pady=1)
        tk.Button(self.ctrl_frame, text="随机一张", command=self._rand_img, width=18).pack(pady=1)
        tk.Button(self.ctrl_frame, text="退出", command=self.master.quit, width=18).pack(pady=1)

        self.log_box = scrolledtext.ScrolledText(self.ctrl_frame, width=36, height=28, font=("Consolas", 9))
        self.log_box.pack(pady=2, fill=tk.BOTH, expand=True)
        self._log("等待训练...")

        self.fig_left.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.06, wspace=0.04, hspace=0.15)
        self.fig_right.subplots_adjust(left=0.04, right=0.96, top=0.97, bottom=0.04, wspace=0.20, hspace=0.30)

    def _log(self, msg):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)
        self.master.update_idletasks()

    def _start_training_thread(self):
        threading.Thread(target=self._train_one_epoch, daemon=True).start()

    def _train_one_epoch(self):
        btn = self.ctrl_frame.winfo_children()[0]
        btn.config(state=tk.DISABLED)
        self._log(f"⏳ 正在训练第 {self.epoch_counter} 个 epoch...")

        train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.testset, batch_size=1000, shuffle=False)
        opt = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        total_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            opt.zero_grad()
            out = self.model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            if (i + 1) % 200 == 0:
                avg_loss = total_loss / (i + 1)
                self._log(f"Step {i+1}, avg_loss={avg_loss:.4f}")

        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                out = self.model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total * 100.0
        self._log(f"✅ 第 {self.epoch_counter} 个 epoch 完成，测试集准确率：{acc:.2f}%\n")

        self.epoch_counter += 1
        btn.config(state=tk.NORMAL, text=f"训练模型 ({self.epoch_counter} epoch)")
        self._update_all_plots()

    def _prev_img(self):
        self.idx = (self.idx - 1) % len(self.trainset)
        self.image, self.label = self.trainset[self.idx]
        self._update_all_plots()

    def _next_img(self):
        self.idx = (self.idx + 1) % len(self.trainset)
        self.image, self.label = self.trainset[self.idx]
        self._update_all_plots()

    def _rand_img(self):
        self.idx = random.randint(0, len(self.trainset) - 1)
        self.image, self.label = self.trainset[self.idx]
        self._update_all_plots()

    def _update_all_plots(self):
        img = self.image.unsqueeze(0).to(self.device)
        img_np = self.image.numpy()[0]
        c1_init = self.c1_init.cpu().numpy()
        c1_trained = self.model.conv1.weight.detach().cpu().numpy()

        with torch.no_grad():
            y_init = F.conv2d(torch.tensor(img_np)[None, None, :, :].float(),
                              torch.tensor(c1_init).float(), bias=None, stride=1, padding=0)
            y_tr = F.conv2d(torch.tensor(img_np)[None, None, :, :].float(),
                            torch.tensor(c1_trained).float(), bias=None, stride=1, padding=0)
            c1_relu = F.relu(F.conv2d(torch.tensor(img_np)[None, None, :, :].float(),
                                      torch.tensor(c1_trained).float(), bias=None, stride=1, padding=0))
            s2_map = F.avg_pool2d(c1_relu, kernel_size=2, stride=2)

        for ax_row in self.axes_left:
            for ax in ax_row:
                ax.clear()
        for i in range(6):
            self._draw_kernel(self.axes_left[i, 0], c1_init[i, 0], f"初始C1核 {i+1}")
            self._imshow_map(self.axes_left[i, 1], y_init[0, i].numpy(), f"初始卷积 {i+1}")
            self._draw_kernel(self.axes_left[i, 2], c1_trained[i, 0], f"训练C1核 {i+1}")
            self._imshow_map(self.axes_left[i, 3], y_tr[0, i].numpy(), f"训练卷积 {i+1}")
            self._imshow_map(self.axes_left[i, 4], s2_map[0, i].numpy(), f"S2池化 {i+1}")
        self.fig_left.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.06, wspace=0.04, hspace=0.15)
        self.canvas_left.draw_idle()

        with torch.no_grad():
            c3_weight = self.model.conv2.weight.detach().cpu()
            c3_out = F.conv2d(s2_map, c3_weight.float(), bias=None, stride=1, padding=0)
            s4_out = F.avg_pool2d(F.relu(c3_out), kernel_size=2, stride=2)

        for ax_row in self.axes_right:
            for ax in ax_row:
                ax.clear()
        c3_kernels_mean = c3_weight.mean(dim=1).numpy()
        for i in range(16):
            self._draw_kernel(self.axes_right[i, 0], c3_kernels_mean[i], f"C3核 {i+1}")
            self._imshow_map(self.axes_right[i, 1], c3_out[0, i].cpu().numpy(), f"C3卷积 {i+1}")
            self._imshow_map(self.axes_right[i, 2], s4_out[0, i].cpu().numpy(), f"S4池化 {i+1}")

        self.fig_right.subplots_adjust(left=0.04, right=0.96, top=0.97, bottom=0.04, wspace=0.20, hspace=0.30)
        self.canvas_right.draw_idle()
        self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))

    def _draw_kernel(self, ax, kernel_2d, title):
        ax.imshow(kernel_2d, cmap="gray", interpolation="nearest", vmin=-0.5, vmax=0.5)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        for r in range(kernel_2d.shape[0]):
            for c in range(kernel_2d.shape[1]):
                v = float(kernel_2d[r, c])
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        color="red" if abs(v) > 0.3 else "black", fontsize=7, weight="bold")

    def _imshow_map(self, ax, arr2d, title):
        a = (arr2d - arr2d.min()) / (arr2d.max() - arr2d.min() + 1e-8)
        ax.imshow(a, cmap="gray", interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

if __name__ == "__main__":
    root = tk.Tk()
    app = LeNet5VisualizerApp(root)
    root.mainloop()
