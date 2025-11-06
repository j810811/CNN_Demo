# -*- coding: utf-8 -*-
"""
LeNet-5 S2→C3 单通道可视化（卷积 + 加和 + 激活）
------------------------------------------------
演示：C3[0] 的生成过程
  S2(0)、S2(1)、S2(2)  →  各自卷积 → 加和 → 加偏置 → tanh 激活
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# ============== 基础函数 ==============
def conv2d(x, k, bias=0.0):
    h, w = x.shape
    kh, kw = k.shape
    oh, ow = h - kh + 1, w - kw + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(x[i:i+kh, j:j+kw] * k) + bias
    return out

def tanh(x):
    return np.tanh(x)

def avg_pool(x, size=2, stride=2):
    h, w = x.shape
    oh, ow = h // stride, w // stride
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            blk = x[i*stride:i*stride+size, j*stride:j*stride+size]
            out[i, j] = np.mean(blk)
    return out

# ============== 数据准备 ==============
tf = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST("./MNIST_data", train=True, download=False, transform=tf)
img, _ = mnist[0]
img = img.squeeze().numpy()  # (28×28)

# ============== C1→S2 ==============
kernels_C1 = [
    np.array([[ 1, 0,-1],
              [ 1, 0,-1],
              [ 1, 0,-1]]),
    np.array([[ 1, 1, 1],
              [ 0, 0, 0],
              [-1,-1,-1]]),
    np.array([[ 0,-1, 0],
              [-1, 4,-1],
              [ 0,-1, 0]]),
    (1/9.0)*np.ones((3,3)),
    np.array([[ 0,-1, 0],
              [-1, 5,-1],
              [ 0,-1, 0]]),
    np.array([[-2,-1, 0],
              [-1, 1, 1],
              [ 0, 1, 2]])
]

S2_maps = []
for k in kernels_C1:
    z = conv2d(img, k)
    a = tanh(z)
    s2 = avg_pool(a, 2, 2)
    s2 = s2[:12, :12]
    S2_maps.append(s2)

# ============== C3 连接表 + 随机卷积核 ==============
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

# ============== 可视化指定 C3 通道的生成过程 ==============
idx = 0  # C3[0]
conns = C3_connection[idx]
kernels = C3_kernels[idx]
bias = np.random.randn() * 0.1

conv_results = []
for c, k in zip(conns, kernels):
    v = conv2d(S2_maps[c], k)
    conv_results.append(v)

sum_map = np.sum(conv_results, axis=0)
sum_bias = sum_map + bias
activated = tanh(sum_bias)

# ============== 绘图 ==============
fig, axes = plt.subplots(3, 5, figsize=(15, 8))
fig.suptitle("LeNet-5：S2→C3[0] 计算过程演示", fontsize=14)

# 第一行：S2 输入
for i, c in enumerate(conns):
    axes[0, i].imshow(S2_maps[c], cmap='gray')
    axes[0, i].set_title(f"S2[{c}]（输入）")
for i in range(len(conns), 5):
    axes[0, i].axis('off')

# 第二行：卷积后
for i, v in enumerate(conv_results):
    axes[1, i].imshow(v, cmap='bwr', vmin=-1, vmax=1)
    axes[1, i].set_title(f"S2[{conns[i]}]*K[{i}]（卷积）")
for i in range(len(conns), 5):
    axes[1, i].axis('off')

# 第三行：加和 → 加偏置 → 激活
axes[2, 0].imshow(sum_map, cmap='bwr', vmin=-1, vmax=1)
axes[2, 0].set_title("卷积结果加和")
axes[2, 1].imshow(sum_bias, cmap='bwr', vmin=-1, vmax=1)
axes[2, 1].set_title("加偏置")
axes[2, 2].imshow(activated, cmap='bwr', vmin=-1, vmax=1)
axes[2, 2].set_title("tanh 激活后输出")
for i in range(3,5):
    axes[2, i].axis('off')

for ax_row in axes:
    for ax in ax_row:
        ax.axis('off')

plt.tight_layout()
plt.show()
