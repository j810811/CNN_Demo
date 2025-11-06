# -*- coding: utf-8 -*-
"""
平均池化教学演示 v4.0
---------------------------------------------
✅ 左图：28×28 原图
✅ 右图：12×12 平均池化结果（红色数值）
✅ 每个像素物理大小一致（右图真实缩小）
✅ 红框同步、平均池化公式动态显示
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision import datasets, transforms

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# === 数据加载 ===
tf = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST("./MNIST_data", train=True, download=False, transform=tf)
img, label = mnist[0]
img = img.squeeze().numpy()

# === 池化参数 ===
size, stride = 2, 2
H, W = img.shape
out_H, out_W = (H - size)//stride + 1, (W - size)//stride + 1
pooled = np.full((out_H, out_W), np.nan)
pos = [0, 0]

# === 图形窗口 ===
fig = plt.figure(figsize=(14, 7))
fig.canvas.manager.set_window_title("平均池化教学演示 v4.0")

# === 比例控制：右图按比例缩小 ===
ratio = out_W / W  # 缩放比 ≈ 0.43
ax_in = fig.add_axes([0.05, 0.25, 0.4, 0.7])
ax_out = fig.add_axes([0.55, 0.25 + (1 - ratio) * 0.35, 0.4 * ratio, 0.7 * ratio])

# === 底部公式区 ===
ax_formula = fig.add_axes([0.05, 0.05, 0.9, 0.12])
ax_formula.axis("off")
text_formula = ax_formula.text(0.5, 0.5, "", fontsize=11, ha="center", va="center", color="blue")

# === 左图 ===
im_in = ax_in.imshow(img, cmap='gray', vmin=0, vmax=1,
                     extent=[0, W, H, 0], interpolation='none')
ax_in.set_title("原图 (28×28)")
ax_in.set_xlim(0, W)
ax_in.set_ylim(H, 0)
ax_in.set_aspect("equal")

# === 右图 ===
im_out = ax_out.imshow(pooled, cmap='gray', vmin=0, vmax=1,
                       extent=[0, out_W, out_H, 0], interpolation='none')
ax_out.set_title("平均池化结果 (12×12)")
ax_out.set_xlim(0, out_W)
ax_out.set_ylim(out_H, 0)
ax_out.set_aspect("equal")

# === 红框 ===
rect_in = Rectangle((0, 0), size, size, edgecolor="red", facecolor="none", lw=2)
rect_out = Rectangle((0, 0), 1, 1, edgecolor="red", facecolor="none", lw=2)
ax_in.add_patch(rect_in)
ax_out.add_patch(rect_out)

# === 数值显示 ===
def show_numbers(ax, data, color="black", fontsize=6):
    for t in list(ax.texts):
        t.remove()
    h, w = data.shape
    for i in range(h):
        for j in range(w):
            if np.isnan(data[i, j]):
                continue
            ax.text(j + 0.5, i + 0.5, f"{data[i,j]:.2f}",
                    ha="center", va="center", color=color, fontsize=fontsize)

def show_numbers_input(ax, data, color="red", fontsize=6):
    for t in list(ax.texts):
        t.remove()
    h, w = data.shape
    for i in range(h):
        for j in range(w):
            ax.text(j + 0.5, i + 0.5, f"{data[i,j]:.1f}",
                    ha="center", va="center", color=color, fontsize=fontsize)

# === 更新显示 ===
def update_display():
    x, y = pos
    rect_in.set_xy((x*stride, y*stride))
    rect_out.set_xy((x, y))

    block = img[y*stride:y*stride+size, x*stride:x*stride+size]
    val = np.mean(block)
    pooled[y, x] = val

    im_out.set_data(pooled)
    show_numbers_input(ax_in, img, color="red")
    show_numbers(ax_out, pooled, color="red")

    # --- 更新标题 ---
    ax_in.set_title(f"原图 (28×28)\n当前位置 ({y},{x}) 平均值={val:.3f}")
    ax_out.set_title(f"平均池化结果 (12×12)\n({y},{x})={val:.3f}")

    # --- 显示平均池化公式 ---
    vals = [f"{block[i,j]:.2f}" for i in range(size) for j in range(size)]
    text_formula.set_text(f"pool[{y},{x}] = ({' + '.join(vals)}) / {size*size} = {val:.3f}")

    fig.canvas.draw_idle()

# === 键盘事件 ===
def on_key(event):
    global pos
    x, y = pos
    if event.key == "right" and x < out_W - 1:
        x += 1
    elif event.key == "left" and x > 0:
        x -= 1
    elif event.key == "down" and y < out_H - 1:
        y += 1
    elif event.key == "up" and y > 0:
        y -= 1
    else:
        return
    pos[:] = [x, y]
    update_display()

fig.canvas.mpl_connect("key_press_event", on_key)
update_display()
plt.show()
