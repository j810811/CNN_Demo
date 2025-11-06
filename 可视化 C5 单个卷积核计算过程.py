# -*- coding: utf-8 -*-
"""
LeNet-5 C5 单个卷积核计算过程可视化（修正版）
- 去除卷积结果 → 输出 的箭头
- Morandi 柔色 + 中文英双语 + 教学版布局
作者：蒋武衡 定制版
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# === 设置中文字体 ===
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# === Morandi 柔色 ===
M_SAGE   = "#A8C3BC"   # 绿色（输入特征图）
M_BLUE   = "#B8C5D6"   # 蓝灰（卷积核）
M_BEIGE  = "#E6D5C3"   # 米色（中间结果）
M_ROSE   = "#D7C4BB"   # 粉灰（偏置）
M_CLAY   = "#C6B8A8"   # 汇总结果
M_CHARCOAL = "#5E6A71" # 深灰线条

# === 图形布局 ===
plt.figure(figsize=(12,6))
ax = plt.gca()
ax.set_xlim(0, 10)
ax.set_ylim(0, 18)
ax.axis("off")

# === 第1列：16个特征图 ===
for i in range(16):
    y = i * 1.0
    rect = Rectangle((0.5, y), 1.0, 0.9, facecolor=M_SAGE, edgecolor=M_CHARCOAL, lw=0.8)
    ax.add_patch(rect)
    ax.text(1.0, y+0.45, f"特征图{i+1}\nFeature{i+1}", fontsize=8, ha='center', va='center')

ax.text(1.0, 16.8, "输入特征图\nInput Feature Maps", ha='center', va='center', fontsize=10, color=M_CHARCOAL)

# === 第2列：16个卷积核 ===
for i in range(16):
    y = i * 1.0
    rect = Rectangle((2.5, y), 1.0, 0.9, facecolor=M_BLUE, edgecolor=M_CHARCOAL, lw=0.8)
    ax.add_patch(rect)
    ax.text(3.0, y+0.45, f"核{i+1}\nKernel{i+1}", fontsize=8, ha='center', va='center')

ax.text(3.0, 16.8, "卷积核\nKernels (5×5)", ha='center', va='center', fontsize=10, color=M_CHARCOAL)

# === 第3列：16个卷积结果标量 ===
for i in range(16):
    y = i * 1.0
    circ = Rectangle((4.5, y+0.3), 0.8, 0.4, facecolor=M_BEIGE, edgecolor=M_CHARCOAL, lw=0.8)
    ax.add_patch(circ)
    ax.text(4.9, y+0.5, f"s{i+1}", fontsize=8, ha='center', va='center')

ax.text(5.0, 16.8, "卷积结果标量\nConv Results", ha='center', va='center', fontsize=10, color=M_CHARCOAL)

# === 第4列：1个偏置项 ===
bias_rect = Rectangle((6.0, 7.5), 1.0, 1.0, facecolor=M_ROSE, edgecolor=M_CHARCOAL, lw=1.0)
ax.add_patch(bias_rect)
ax.text(6.5, 8.0, "偏置\nBias b", fontsize=9, ha='center', va='center')
ax.text(6.5, 16.8, "偏置项\nBias", ha='center', va='center', fontsize=10, color=M_CHARCOAL)

# === 第5列：汇总输出 ===
out_rect = Rectangle((8.0, 7.5), 1.0, 1.0, facecolor=M_CLAY, edgecolor=M_CHARCOAL, lw=1.0)
ax.add_patch(out_rect)
ax.text(8.5, 8.0, "输出\nOutput y", fontsize=9, ha='center', va='center')
ax.text(8.5, 16.8, "卷积核输出\nC5 Output", ha='center', va='center', fontsize=10, color=M_CHARCOAL)

# === 箭头连接 ===
for i in range(16):
    y = i * 1.0 + 0.45
    # 特征图 -> 卷积核
    ax.add_patch(FancyArrowPatch((1.5, y+0.05), (2.5, y+0.05),
                                 arrowstyle='->', color=M_CHARCOAL, lw=0.7))
    # 卷积核 -> 卷积结果
    ax.add_patch(FancyArrowPatch((3.5, y+0.05), (4.5, y+0.05),
                                 arrowstyle='->', color=M_CHARCOAL, lw=0.7))
    # 卷积结果 -> 偏置
    ax.add_patch(FancyArrowPatch((5.3, y), (6.0, 8.0),
                                 arrowstyle='-', color=M_CHARCOAL, lw=0.5, alpha=0.5))

# 偏置 -> 输出
ax.add_patch(FancyArrowPatch((7.0, 8.0), (8.0, 8.0),
                             arrowstyle='->', color=M_CHARCOAL, lw=1.0))

# === 汇总公式 ===
ax.text(5.3, 4.0, r"$y = \sum_{k=1}^{16}s_k + b$", fontsize=11, color=M_CHARCOAL)

# === 标题 ===
plt.title("LeNet-5：C5 单卷积核计算过程（Morandi）\nLeNet-5: Single Kernel Computation in C5 Layer (Morandi)",
          fontsize=12, color=M_CHARCOAL, pad=20)

plt.tight_layout()
plt.show()
