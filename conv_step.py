# -*- coding: utf-8 -*-
"""
卷积教学动画 v4（双红框 + 完整公式 + bwr色图 + 自动换行）
--------------------------------------------------
→ 右移一步：计算一个输出像素
↓ 下一行：从新行开始
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision import datasets, transforms
import textwrap

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# === 加载 MNIST 数据 ===
tf = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST("./MNIST_data", train=True, download=False, transform=tf)
img, label = mnist[0]
img = img.squeeze().numpy()
kernel = np.random.randn(5, 5)
out = np.full((24, 24), np.nan)

# === 图布局 ===
fig, axes = plt.subplots(
    1, 5, figsize=(15, 4),
    gridspec_kw={'width_ratios': [28, 2, 5, 2, 24]}
)
ax_in, ax_space1, ax_k, ax_space2, ax_out = axes
ax_space1.axis('off')
ax_space2.axis('off')

# === 通用绘制函数 ===
def show_img(ax, data, title, cmap='gray', vmin=None, vmax=None):
    h, w = data.shape
    im = ax.imshow(data, cmap=cmap, extent=[-0.5, w-0.5, h-0.5, -0.5], vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(0, w, 1))
    ax.set_yticks(np.arange(0, h, 1))
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which='minor', color='#777', linewidth=0.3, alpha=0.6)
    ax.grid(which='major', visible=False)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_title(title, fontsize=11)
    return im

def show_numbers(ax, data):
    h, w = data.shape
    for i in range(h):
        for j in range(w):
            ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center",
                    fontsize=6, color="red")

# === 初始显示 ===
show_img(ax_in, img, "原图 (28×28)", cmap='gray', vmin=0, vmax=1)
show_numbers(ax_in, img)
im_k = show_img(ax_k, kernel, "卷积核 (5×5)", cmap='gray')
show_numbers(ax_k, kernel)
im_out = show_img(ax_out, np.zeros_like(out), "输出 (24×24)", cmap='bwr', vmin=-2, vmax=2)

# === 红框 ===
rect_in = Rectangle((-0.5, -0.5), 5, 5, edgecolor="red", facecolor="none", lw=1)
rect_out = Rectangle((-0.5, -0.5), 1, 1, edgecolor="red", facecolor="none", lw=1)
ax_in.add_patch(rect_in)
ax_out.add_patch(rect_out)

# === 卷积函数 ===
def conv_step(img, kernel, x, y):
    region = img[y:y+5, x:x+5]
    return np.sum(region * kernel), region

# === 底部公式区 ===
fig.subplots_adjust(bottom=0.28)
ax_formula = fig.add_axes([0.05, -0.10, 0.9, 0.25])
ax_formula.axis('off')
text_formula = ax_formula.text(0.5, 0.5, "", fontsize=10, ha="center", va="center", color="blue")

# === 键盘事件 ===
pos = [0, 0]
def on_key(event):
    global pos, out
    x, y = pos
    if event.key == 'right':
        if x < 23:
            x += 1
    elif event.key == 'down':
        if y < 23:
            y += 1
            x = 0
    else:
        return
    pos[:] = [x, y]

    # 红框移动
    rect_in.set_xy((x - 0.5, y - 0.5))
    rect_out.set_xy((x - 0.5, y - 0.5))

    # 卷积计算
    value, region = conv_step(img, kernel, x, y)
    out[y, x] = value

    # 更新输出色图
    im_out.set_data(out)
    finite_vals = out[np.isfinite(out)]
    if finite_vals.size > 0:
        im_out.set_clim(vmin=np.nanmin(finite_vals), vmax=np.nanmax(finite_vals))

    # 更新输出文字（仅显示已计算区域）
    for t in list(ax_out.texts):
        t.remove()
    for i in range(y+1):
        max_j = 24 if i < y else x+1
        for j in range(max_j):
            if not np.isnan(out[i,j]):
                ax_out.text(j, i, f"{out[i,j]:.1f}", ha="center", va="center",
                            fontsize=6, color="red")

    # === 展开公式 ===
    products = [f"{region[i,j]:.1f}×{kernel[i,j]:.1f}" for i in range(5) for j in range(5)]
    full_formula = " + ".join(products) + f" = {value:.3f}"
    # 自动换行（每行约 100 字符）
    wrapped = textwrap.fill(full_formula, width=100)
    text_formula.set_text(f"y[{y},{x}] = {wrapped}")

    fig.canvas.draw_idle()

fig.canvas.mpl_connect("key_press_event", on_key)
plt.tight_layout()
plt.show()
