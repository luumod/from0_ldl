import math
import torch
import random

'''
w1 w2 w3 
b
'''


# 生成随机数据
examples = 10000
true_w = torch.tensor([2.1, -3.2, 4.2])
true_b = 2.989
def synthetic_data(w, b ,num_examples):
    x = torch.normal(0, 1, (num_examples, len(w))) # 10000 * 3
    y = torch.matmul(x, w) + b
    y += torch.normal(0 ,0.01, y.shape)
    return x, y.reshape((-1, 1))

features, labels = synthetic_data(true_w, true_b, examples)

# 获取小批量的数据
def get_batch(batch_size, features, labels):
    num_examples = len(features)
    index = list(range(num_examples))
    random.shuffle(index)
    for i in range(0, num_examples, batch_size):
        batch_index = torch.tensor(index[i:min(i+batch_size, num_examples)])
        yield features[batch_index], labels[batch_index]

batch_size = 30
for X, y in get_batch(batch_size, features, labels):
    print(X, '\n', y)
    break

# 定义模型
def model(X, w, b):
    return torch.matmul(X, w.T) + b # 10000 * 1

# 定义损失函数
def loss(y_hat, y):
    return (y_hat.reshape(y.shape) - y) ** 2 / 2

# 定义优化器
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 初始化参数
w = torch.normal(0, 0.01, (1, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 训练
lr = 0.01
epoches = 3
for epoch in range(epoches):
    for X, y in get_batch(batch_size, features, labels):
        y_hat = model(X, w, b)
        l = loss(y_hat, y)
        l.sum().backward() # 10000*1 二维变一维
        sgd([w, b], lr, batch_size) # optim.step()
    with torch.no_grad():
        train_y = model(features, w, b)
        train_loss = loss(train_y, labels)
        print(f'epoch: {epoch + 1} | loss: {train_loss.mean()}')