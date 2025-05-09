import torch
import torch.nn as nn

# 卷积运算
def corr2d(X, K):
    '''
    :param X: 输入张量
    :param K: 卷积核张量
    :return: Y 输出张量
    '''
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

X = torch.arange(16).reshape(4, 4)
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))


# 卷积层
class Conv2d(nn.Module):
    def __init__(self, kernal_size):
        super(Conv2d, self).__init__()
        self.weight = nn.Parameter(torch.rand(kernal_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias

# 边缘检测
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)

K = torch.tensor([[1.0, -1.0]])
# 可以检测水平边缘
Y = corr2d(X, K)
print(Y)

# 无法检测垂直边缘
print(corr2d(X.T, K))

# 学习卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
X = X.reshape(1, 1, 6, 8) # 初始输入
Y = Y.reshape(1, 1, 6, 7) # 经卷积后的输出
lr = 3e-2

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch: {i + 1}, loss:{l.sum():.3f}')

print(conv2d.weight.data.reshape((1,2)))


