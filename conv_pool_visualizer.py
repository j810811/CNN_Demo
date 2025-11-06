# -*- coding: utf-8 -*-
"""
LeNet-5 S2→C3 教学可视化（竖向居中 + 点击高亮）
------------------------------------------------
列布局：
  第1列：原始 MNIST（28×28）
  第2列：S2（6 张 12×12）
  第3列：C3（16 张 8×8）
交互：
  点击右列 C3 → 高亮对应的 S2 图
  点击空白 → 取消高亮
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ============== 基础函数 ==============
def conv2d(x, k, bias=0.0):
    h, w = x.shape
    kh, kw = k.shape
    oh, ow = h - kh + 1, w - kw + 1
    out = np.zeros((oh, ow), dtype=float)
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(x[i:i+kh, j:j+kw] * k) + bias
    return out

def tanh(x):
    return np.tanh(x)

def avg_pool(x, size=2, stride=2):
    h, w = x.shape
    oh, ow = h // stride, w // stride
    out = np.zeros((oh, ow), dtype=float)
    for i in range(oh):
        for j in range(ow):
            blk = x[i*stride:i*stride+size, j*stride:j*stride+size]
            out[i, j] = np.mean(blk)
    return out

def put_numbers(ax, arr, color="black", fs=5):
    h, w = arr.shape
    for i in range(h):
        for j in range(w):
            ax.text(j + 0.5, i + 0.5, f"{arr[i, j]:.2f}",
                    ha="center", va="center", fontsize=fs, color=color)

# ============== 数据加载 ==============
tf = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST("./MNIST_data", train=True, download=False, transform=tf)
img, _ = mnist[0]
img = img.squeeze().numpy()  # (28,28)

# ============== C1→S2 ==============
kernels_C1 = [
    np.array([[ 1, 0,-1],
              [ 1, 0,-1],
              [ 1, 0,-1]]),           # 垂直边缘
    np.array([[ 1, 1, 1],
              [ 0, 0, 0],
              [-1,-1,-1]]),           # 水平边缘
    np.array([[ 0,-1, 0],
              [-1, 4,-1],
              [ 0,-1, 0]]),           # Laplacian
    (1/9.0)*np.ones((3,3)),           # 均值
    np.array([[ 0,-1, 0],
              [-1, 5,-1],
              [ 0,-1, 0]]),           # 锐化
    np.array([[-2,-1, 0],
              [-1, 1, 1],
              [ 0, 1, 2]])            # 浮雕
]

S2_maps = []
for k in kernels_C1:
    z = conv2d(img, k)
    a = tanh(z)
    s2 = avg_pool(a, 2, 2)
    s2 = s2[:12, :12]
    S2_maps.append(s2)
S2_maps = [np.asarray(m) for m in S2_maps]

# ============== C3：论文连接表 ==============
C3_connection = [
    [0,1,2], [1,2,3], [2,3,4], [3,4,5],
    [0,1,3,5], [0,2,4,5], [1,3,5], [0,2,3,4],
    [2,4,5], [1,2,5], [0,1,4], [1,2,3,4],
    [0,3,5], [0,2,5], [1,4,5], [0,1,2,3,4,5]
]

np.random.seed(0)
C3_kernels = []
for conns in C3_connection:
    C3_kernels.append([np.random.randn(5,5)*0.2 for _ in conns])

C3_maps = []
for idx, conns in enumerate(C3_connection):
    acc = None
    for c, k in zip(conns, C3_kernels[idx]):
        v = conv2d(S2_maps[c], k)
        acc = v if acc is None else acc + v
    C3_maps.append(tanh(acc))

# ============== Tk + 滚动画布 ==============
root = tk.Tk()
root.title("LeNet-5 S2→C3 教学可视化（竖向居中 + 点击高亮）")

main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

cnv = tk.Canvas(main_frame)
sb_y = ttk.Scrollbar(main_frame, orient="vertical", command=cnv.yview)
sb_x = ttk.Scrollbar(main_frame, orient="horizontal", command=cnv.xview)
container = ttk.Frame(cnv)

container.bind("<Configure>", lambda e: cnv.configure(scrollregion=cnv.bbox("all")))
cnv.create_window((0, 0), window=container, anchor="nw")
cnv.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)

cnv.pack(side="left", fill="both", expand=True)
sb_y.pack(side="right", fill="y")
sb_x.pack(side="bottom", fill="x")

# ============== 像素等比布局 ==============
TOTAL_UNITS_H = 128.0
MARGIN_L, MARGIN_R = 0.04, 0.02
GAP_X = 0.04
COL_UNIT_W = 28 + 12 + 8
W_LEFT  = 28.0 / COL_UNIT_W * (1.0 - MARGIN_L - MARGIN_R - 2*GAP_X)
W_MID   = 12.0 / COL_UNIT_W * (1.0 - MARGIN_L - MARGIN_R - 2*GAP_X)
W_RIGHT =  8.0 / COL_UNIT_W * (1.0 - MARGIN_L - MARGIN_R - 2*GAP_X)
X_LEFT  = MARGIN_L
X_MID   = X_LEFT  + W_LEFT  + GAP_X
X_RIGHT = X_MID   + W_MID   + GAP_X
UNIT_H = (1.0 - 0.04 - 0.02) / TOTAL_UNITS_H
Y_TOP = 1.0 - 0.04

fig = plt.Figure(figsize=(12, 30), dpi=100)

# --- 交互数据容器 ---
S2_axes = []
C3_axes = []

# === 三列居中对齐 + 调整图间距 ===
GAP_Y_S2 = 2.0 * UNIT_H
GAP_Y_C3 = 1.0 * UNIT_H
H_LEFT = 28.0
H_MID = 6 * 12.0 + (6 - 1) * (GAP_Y_S2 / UNIT_H)
H_RIGHT = 16 * 8.0 + (16 - 1) * (GAP_Y_C3 / UNIT_H)
H_MAX = max(H_LEFT, H_MID, H_RIGHT)
Y_CENTER = 0.5 * (1.0 + 0.04 - 0.02)
Y_BASE = Y_CENTER + (H_MAX / 2) * UNIT_H

# --- 第 1 列 ---
cur_y = Y_BASE - (H_MAX - H_LEFT) / 2 * UNIT_H
ax = fig.add_axes([X_LEFT, cur_y - 28 * UNIT_H, W_LEFT, 28 * UNIT_H])
ax.imshow(img, cmap='gray', extent=[0, 28, 28, 0])
ax.set_aspect('equal')
ax.set_title("原始 MNIST（28×28）", fontsize=11)
ax.axis('off')

# --- 第 2 列 ---
cur_y = Y_BASE - (H_MAX - H_MID) / 2 * UNIT_H
for i, m in enumerate(S2_maps):
    h_units = 12.0
    ax = fig.add_axes([X_MID, cur_y - h_units * UNIT_H, W_MID, h_units * UNIT_H])
    ax.imshow(m, cmap='bwr', vmin=-1, vmax=1, extent=[0, 12, 12, 0])
    ax.set_aspect('equal')
    ax.set_title(f"S2 特征图 {i}（12×12）", fontsize=10)
    ax.axis('off')
    put_numbers(ax, m, color="black", fs=5)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_edgecolor("none")  # 默认无框
    S2_axes.append(ax)
    cur_y -= (h_units * UNIT_H + GAP_Y_S2)

# --- 第 3 列 ---
cur_y = Y_BASE - (H_MAX - H_RIGHT) / 2 * UNIT_H
for i, m in enumerate(C3_maps):
    h_units = 8.0
    ax = fig.add_axes([X_RIGHT, cur_y - h_units * UNIT_H, W_RIGHT, h_units * UNIT_H])
    ax.imshow(m, cmap='bwr', vmin=-1, vmax=1, extent=[0, 8, 8, 0])
    ax.set_aspect('equal')
    ax.set_title(f"C3 特征图 {i}（8×8）\n连接：{C3_connection[i]}", fontsize=9)
    ax.axis('off')
    put_numbers(ax, m, color="black", fs=5)
    C3_axes.append((ax, i))
    cur_y -= (h_units * UNIT_H + GAP_Y_C3)

# === 点击事件：高亮对应的 S2 图 ===
def on_click(event):
    if event.inaxes is None:
        for a in S2_axes:
            for spine in a.spines.values():
                spine.set_edgecolor("none")
        canvas_fig.draw()
        return
    for ax, idx in C3_axes:
        if event.inaxes == ax:
            conns = C3_connection[idx]
            for a in S2_axes:
                for spine in a.spines.values():
                    spine.set_edgecolor("none")
            for c in conns:
                s2_ax = S2_axes[c]
                for spine in s2_ax.spines.values():
                    spine.set_linewidth(2.5)
                    spine.set_edgecolor("red")
            canvas_fig.draw()
            return

# === 嵌入 Tk ===
canvas_fig = FigureCanvasTkAgg(fig, master=container)
canvas_fig.draw()
fig.canvas.mpl_connect('button_press_event', on_click)
canvas_fig.get_tk_widget().pack(fill="both", expand=True)

root.mainloop()
