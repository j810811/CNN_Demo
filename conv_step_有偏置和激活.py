# -*- coding: utf-8 -*-
"""
卷积教学动画 v6.8
---------------------------------------------
✅ 输出图像像素值为黑色文字
✅ 空格键切换激活函数：tanh ↔ sigmoid
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

# === 初始为 tanh ===
current_activation = "tanh"
bias = 0.1

def tanh(x): return np.tanh(x)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def activation(x): return tanh(x) if current_activation == "tanh" else sigmoid(x)

# === 主布局 ===
fig, axes = plt.subplots(1, 5, figsize=(15, 5), gridspec_kw={'width_ratios': [28, 2, 5, 2, 24]})
ax_in, ax_space1, ax_k_dummy, ax_space2, ax_out = axes
ax_space1.axis('off'); ax_space2.axis('off')

# === 手动创建卷积核及附属区域 ===
left = ax_k_dummy.get_position().x0
width = ax_k_dummy.get_position().width
ax_k = fig.add_axes([left, 0.78, width, 0.18])
ax_bias = fig.add_axes([left, 0.70, width, 0.04])
ax_formula_text = fig.add_axes([left, 0.63, width, 0.05])
ax_actplot = fig.add_axes([left, 0.38, width, 0.22])
ax_k_dummy.remove()
for ax in [ax_bias, ax_formula_text]:
    ax.axis("off")

# === 激活函数曲线绘制 ===
def draw_activation_curve():
    ax_actplot.clear()
    x_vals = np.linspace(-3, 3, 400)
    if current_activation == "tanh":
        y_vals = tanh(x_vals)
        title = "激活函数曲线：tanh(z)"
        y_lim = (-1.1, 1.1)
    else:
        y_vals = sigmoid(x_vals)
        title = "激活函数曲线：sigmoid(z)"
        y_lim = (-0.1, 1.1)
    ax_actplot.plot(x_vals, y_vals, color="blue", lw=1.5)
    ax_actplot.axhline(0, color="gray", lw=0.5)
    ax_actplot.axvline(0, color="gray", lw=0.5)
    ax_actplot.set_title(title, fontsize=9)
    ax_actplot.set_xlim(-3, 3)
    ax_actplot.set_ylim(*y_lim)
    return ax_actplot.plot([0], [0], 'ro', markersize=5)[0]

act_point = draw_activation_curve()

# === 通用绘图函数 ===
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

def show_numbers(ax, data, color="red"):
    """通用数字显示函数"""
    h, w = data.shape
    for i in range(h):
        for j in range(w):
            ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center",
                    fontsize=6, color=color)

# === 初始显示 ===
show_img(ax_in, img, "原图 (28×28)", cmap='gray', vmin=0, vmax=1)
show_numbers(ax_in, img, color="red")
im_k = show_img(ax_k, kernel, "卷积核 (5×5)", cmap='gray')
show_numbers(ax_k, kernel, color="red")
im_out = show_img(ax_out, np.zeros_like(out), "激活后输出 (24×24)", cmap='bwr', vmin=-1, vmax=1)

# === 红框 ===
rect_in = Rectangle((-0.5, -0.5), 5, 5, edgecolor="red", facecolor="none", lw=1)
rect_out = Rectangle((-0.5, -0.5), 1, 1, edgecolor="red", facecolor="none", lw=1)
ax_in.add_patch(rect_in)
ax_out.add_patch(rect_out)

# === 卷积计算 ===
def conv_step(img, kernel, x, y):
    region = img[y:y+5, x:x+5]
    z = np.sum(region * kernel) + bias
    a = activation(z)
    return z, a, region

# === 底部公式 ===
fig.subplots_adjust(bottom=0.3)
ax_formula = fig.add_axes([0.05, -0.10, 0.9, 0.25])
ax_formula.axis('off')
text_formula = ax_formula.text(0.5, 0.5, "", fontsize=10, ha="center", va="center", color="blue")

# === 键盘事件 ===
pos = [0, 0]
def on_key(event):
    global pos, out, current_activation, act_point

    # 空格：切换激活函数
    if event.key == " ":
        current_activation = "sigmoid" if current_activation == "tanh" else "tanh"
        act_point = draw_activation_curve()
        fig.canvas.draw_idle()
        return

    # 移动卷积窗口
    x, y = pos
    if event.key == 'right' and x < 23:
        x += 1
    elif event.key == 'down' and y < 23:
        y += 1
        x = 0
    else:
        return
    pos[:] = [x, y]

    rect_in.set_xy((x - 0.5, y - 0.5))
    rect_out.set_xy((x - 0.5, y - 0.5))

    z, a, region = conv_step(img, kernel, x, y)
    out[y, x] = a

    im_out.set_data(out)
    finite_vals = out[np.isfinite(out)]
    if finite_vals.size > 0:
        im_out.set_clim(vmin=np.nanmin(finite_vals), vmax=np.nanmax(finite_vals))

    # 清空并重新绘制输出图像数值（⚠️黑色）
    for t in list(ax_out.texts):
        t.remove()
    for i in range(y+1):
        max_j = 24 if i < y else x+1
        for j in range(max_j):
            if not np.isnan(out[i,j]):
                ax_out.text(j, i, f"{out[i,j]:.2f}", ha="center", va="center",
                            fontsize=6, color="black")  # ← 改为黑色

    # 下方显示更新
    ax_bias.clear(); ax_bias.axis("off")
    ax_bias.text(0.5, 0.5, f"偏置 b = {bias:.2f}", ha="center", va="center", fontsize=9, color="purple")

    ax_formula_text.clear(); ax_formula_text.axis("off")
    if current_activation == "tanh":
        formula_math = (r"$a = f(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$"
                        + f"\n z={z:.2f}, a={a:.3f}")
    else:
        formula_math = (r"$a = f(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$"
                        + f"\n z={z:.2f}, a={a:.3f}")
    ax_formula_text.text(0.5, 0.5, formula_math, ha="center", va="center",
                         fontsize=9, color="green")

    act_point.set_data([z], [a])
    ax_actplot.set_title(f"激活函数曲线：{current_activation}\n z={z:.2f} → a={a:.2f}", fontsize=9)

    # 底部公式更新
    products = [f"{region[i,j]:.1f}×{kernel[i,j]:.1f}" for i in range(5) for j in range(5)]
    full_formula = " + ".join(products) + f" + {bias:.1f} → {current_activation}(...) = {a:.3f}"
    wrapped = textwrap.fill(full_formula, width=100)
    text_formula.set_text(f"y[{y},{x}] = {wrapped}")

    fig.canvas.draw_idle()

fig.canvas.mpl_connect("key_press_event", on_key)
plt.tight_layout()
plt.show()
