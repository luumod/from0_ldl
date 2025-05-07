# import math
# from timer import Timer
# import numpy as np
# import pandas as pd
# import torch
#
# n = 10000
# a = torch.ones([n])
# b = torch.ones([n])
# print(a)
# print(b)
#
# c = torch.zeros(n)
# timer = Timer()
# for i in range(n):
#     c[i] = a[i] + b[i]
# print(f'{timer.stop()} sec') # 0.1 s
#
# timer.start()
# d = a + b
# print(f'{timer.stop()} sec')# 0 s
#
# def normal(x, mu, sigma):
#     p = 1 / (math.sqrt(2 * math.pi * sigma ** 2))
#     return p * math.exp(-0.5 / sigma**2 * (x - mu) ** 2)
#
# x = np.arange(-7,7,0.1)
# params = [(0,1), (0,2), (0,3)]

# 从零实现线性回归
# import random
# import torch
#
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w))) # 1000 * 2
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
#
# true_w = torch.tensor([2, -3.4]) # 1 * 2
# true_b = 4.2
# features, labels = synthetic_data(true_w, true_b, 1000)
#
# def data_iter(batch_size, features, labels):
#     num_examples = len(features)
#     indices = list(range(num_examples))
#     random.shuffle(indices)
#     for i in range(0, num_examples, batch_size):
#         batch_indices = torch.tensor(
#             indices[i:min(i + batch_size, num_examples)]
#         )
#         yield features[batch_indices], labels[batch_indices]
#
# batch_size = 10
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break
#
# w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
# b = torch.zeros(1, requires_grad=True)
#
# def linreg(X, w, b):
#     return torch.matmul(X, w) + b
#
# def square_loss(y_hat, y):
#     return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
#
# def sgd(params, lr, batch_size):
#     with torch.no_grad():
#         for param in params:
#             param -= lr * param.grad / batch_size
#             param.grad.zero_()
#
#
# lr = 0.03
# num_epochs = 3
# model = linreg
# loss = square_loss
#
# for epoch in range(num_epochs):
#     # 训练
#     for X, y in data_iter(batch_size, features, labels):
#         y_hat = model(X, w, b)
#         l = loss(y_hat, y)
#         l.sum().backward()
#         sgd([w,b], lr, batch_size)
#     # 评估
#     with torch.no_grad():
#         train_l = loss(model(features, w, b), labels)
#         print(f'epoch {epoch + 1}, loss: {float(train_l.mean()):f}')
#
#
# print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
# print(f'b的估计误差：{true_b - b}')


import numpy as np
import torch
from torch.utils import data
import torch.nn as nn

true_w = torch.tensor([2,-9.3])
true_b = 9.43
features, labels = synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

model = nn.Sequential(nn.Linear(2, 1))

model[0].weight.data.normal_(0, 0.01)
model[0].bias.data.fill_(0)

loss = nn.MSELoss()

optim = torch.optim.SGD(model.parameters(), lr=0.03)

num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter:
        optim.zero_grad()
        y_hat = model(X)
        l = loss(y_hat, y)
        l.backward()
        optim.step()
    l = loss(model(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = model[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = model[0].bias.data
print('b的估计误差：', true_b - b)