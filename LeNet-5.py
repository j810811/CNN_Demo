# -*- coding: utf-8 -*-
"""
LeNet-5 (NumPy 实现，可训练版 + 卷积核可视化)
作者：蒋武衡（教学版）
"""

import numpy as np
import gzip
import matplotlib.pyplot as plt

# ============================================================
# 一、MNIST 数据加载
# ============================================================
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(16)
        data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32)
        data = data.reshape(-1, 1, 28, 28) / 255.0
        return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

x_train = load_mnist_images('./MNIST_data/train-images-idx3-ubyte.gz')[:1000]
y_train = load_mnist_labels('./MNIST_data/train-labels-idx1-ubyte.gz')[:1000]
y_train_oh = one_hot(y_train)

x_test = load_mnist_images('./MNIST_data/t10k-images-idx3-ubyte.gz')[:200]
y_test = load_mnist_labels('./MNIST_data/t10k-labels-idx1-ubyte.gz')[:200]

# ============================================================
# 二、基本函数（卷积 / 池化 / 激活）
# ============================================================
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def conv2d(x, w, b, stride=1):
    n, c, h, w_in = x.shape
    out_c, _, k, _ = w.shape
    h_out = (h - k)//stride + 1
    w_out = (w_in - k)//stride + 1
    out = np.zeros((n, out_c, h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            region = x[:, :, i*stride:i*stride+k, j*stride:j*stride+k]
            out[:, :, i, j] = np.tensordot(region, w, axes=([1,2,3],[1,2,3])) + b
    return out

def conv2d_back(dout, x, w, stride=1):
    """
    卷积层反向传播
    dout: [N, out_c, H_out, W_out]
    x:    [N, in_c, H, W]
    w:    [out_c, in_c, K, K]
    返回:
        dx: 与 x 同形
        dw: 与 w 同形
        db: [out_c]
    """
    N, in_c, H, W = x.shape
    out_c, _, K, _ = w.shape
    _, _, H_out, W_out = dout.shape

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.sum(dout, axis=(0, 2, 3))

    for n in range(N):
        for oc in range(out_c):
            for i in range(H_out):
                for j in range(W_out):
                    # dout[n, oc, i, j] 是标量
                    dw[oc] += dout[n, oc, i, j] * x[n, :, i*stride:i*stride+K, j*stride:j*stride+K]
                    dx[n, :, i*stride:i*stride+K, j*stride:j*stride+K] += dout[n, oc, i, j] * w[oc]

    return dx, dw, db


def avg_pool2d(x, size=2, stride=2):
    n, c, h, w = x.shape
    h_out = (h - size)//stride + 1
    w_out = (w - size)//stride + 1
    out = np.zeros((n, c, h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            out[:, :, i, j] = np.mean(x[:, :, i*stride:i*stride+size, j*stride:j*stride+size], axis=(2,3))
    return out

def avg_pool2d_back(dout, x, size=2, stride=2):
    n, c, h, w = x.shape
    dx = np.zeros_like(x)
    h_out, w_out = dout.shape[2], dout.shape[3]
    for i in range(h_out):
        for j in range(w_out):
            dx[:, :, i*stride:i*stride+size, j*stride:j*stride+size] += dout[:, :, i:i+1, j:j+1] / (size*size)
    return dx

# ============================================================
# 三、参数初始化
# ============================================================
np.random.seed(42)
W1 = np.random.randn(6, 1, 5, 5) * np.sqrt(2/25)
b1 = np.zeros((6,))
W2 = np.random.randn(16, 6, 5, 5) * np.sqrt(2/(6*25))
b2 = np.zeros((16,))
W3 = np.random.randn(256, 120) * np.sqrt(2/256)
b3 = np.zeros((120,))
W4 = np.random.randn(120, 84) * np.sqrt(2/120)
b4 = np.zeros((84,))
W5 = np.random.randn(84, 10) * np.sqrt(2/84)
b5 = np.zeros((10,))

# ============================================================
# 四、训练函数（带反向传播）
# ============================================================
def forward(x):
    z1 = conv2d(x, W1, b1); a1 = relu(z1)
    p1 = avg_pool2d(a1)
    z2 = conv2d(p1, W2, b2); a2 = relu(z2)
    p2 = avg_pool2d(a2)
    flat = p2.reshape(x.shape[0], -1)
    z3 = flat @ W3 + b3; a3 = relu(z3)
    z4 = a3 @ W4 + b4; a4 = relu(z4)
    z5 = a4 @ W5 + b5
    out = softmax(z5)
    cache = (x, z1, a1, p1, z2, a2, p2, flat, a3, a4)
    return out, cache

def backward(y_true, out, cache, lr=0.01):
    global W1,b1,W2,b2,W3,b3,W4,b4,W5,b5
    x, z1, a1, p1, z2, a2, p2, flat, a3, a4 = cache
    m = y_true.shape[0]

    dz5 = (out - y_true)/m
    dW5 = a4.T @ dz5; db5 = np.sum(dz5, axis=0)
    da4 = dz5 @ W5.T

    dz4 = da4 * relu_deriv(a3 @ W4 + b4)
    dW4 = a3.T @ dz4; db4 = np.sum(dz4, axis=0)
    da3 = dz4 @ W4.T

    dz3 = da3 * relu_deriv(flat @ W3 + b3)
    dW3 = flat.T @ dz3; db3 = np.sum(dz3, axis=0)
    dflat = dz3 @ W3.T
    dp2 = dflat.reshape(p2.shape)

    da2 = avg_pool2d_back(dp2, a2)
    dz2 = da2 * relu_deriv(z2)
    dp1, dW2, db2 = conv2d_back(dz2, p1, W2)
    da1 = avg_pool2d_back(dp1, a1)
    dz1 = da1 * relu_deriv(z1)
    dx, dW1, db1 = conv2d_back(dz1, x, W1)

    # 更新参数
    for W,dW,b,db in [(W5,dW5,b5,db5),(W4,dW4,b4,db4),
                      (W3,dW3,b3,db3),(W2,dW2,b2,db2),(W1,dW1,b1,db1)]:
        W -= lr*dW; b -= lr*db

def train(epochs=5, lr=0.01):
    for epoch in range(epochs):
        out, cache = forward(x_train)
        loss = -np.mean(np.sum(y_train_oh*np.log(out+1e-8), axis=1))
        backward(y_train_oh, out, cache, lr)
        print(f"Epoch {epoch+1}/{epochs}, Loss={loss:.4f}")
        visualize_kernels(W1, title=f"Epoch {epoch+1}")

# ============================================================
# 五、卷积核可视化
# ============================================================
def visualize_kernels(W, title="Conv1 Kernels"):
    plt.figure(figsize=(6,1.5))
    for i in range(W.shape[0]):
        plt.subplot(1, W.shape[0], i+1)
        plt.imshow(W[i,0,:,:], cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# ============================================================
# 六、执行训练与评估
# ============================================================
train(epochs=5, lr=0.005)

def accuracy(x, y):
    preds = np.argmax(forward(x)[0], axis=1)
    return np.mean(preds == y)

print("✅ Test Accuracy:", accuracy(x_test, y_test))
