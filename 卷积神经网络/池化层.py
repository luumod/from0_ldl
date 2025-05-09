import torch
import torch.nn as nn

def pool2d(X, pool_size, mode='max'):
    '''
    :param X: 输入矩阵
    :param pool_size: 池化层的尺寸
    :param mode: 最大池化层 / 平均池化层
    '''
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i+p_h, j:j+p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i+p_h, j:j+p_w].mean()
    return Y

X = torch.tensor([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0],[7.0, 8.0, 9.0]])
print(pool2d(X, (2, 2)))

# 填充与步幅
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
pool2d = nn.MaxPool2d(3) # 步幅stride与池化层的窗口大小相同
print(pool2d(X))

pool2d = nn.MaxPool2d(3, padding=1, stride=2) # 自定义设置步幅与填充
print(pool2d(X))

pool2d = nn.MaxPool2d((2, 3), padding=(0, 1), stride=(2, 3))

# 多通道
X = torch.cat((X, X + 1), 1)
print(X.shape)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X).shape)