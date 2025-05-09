import torch
import torch.nn as nn

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

def corr2d_multi_in(X, K):
    return sum(corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]]) # 2*2*2

z = corr2d_multi_in(X, K)

print(X.shape)
print(K.shape)
print(z.shape)


def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

K = torch.stack((K, K + 1, K + 2), 0) # 3 * 2*2*2 ：输出三通道
print(K.shape)

z = corr2d_multi_in_out(X, K)