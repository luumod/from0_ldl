import torch
import random

# 设置一个真实的w与b
true_w = torch.tensor([4.2, -2.9])
true_b = 9.7

# 生成随机数据
def gen_data(w, b, num_examples):
    '''
    :return:
    1. num_examples*特征数 的矩阵
    2. num_examples*1 的矩阵
    '''
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.02, y.shape)
    return X, y.reshape(-1,1)

features, labels = gen_data(true_w, true_b, 1000)

# 批量化处理
def get_batch(features, labels, batch_size):
    num_examples = len(features)
    index = list(range(num_examples))
    random.shuffle(index)
    for i in range(0, num_examples, batch_size):
        ii = torch.tensor(index[i:min(i+batch_size, num_examples)])
        yield features[ii], labels[ii]

# 随机生成w与b
w = torch.normal(0, 0.01, (1,2), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义模型
def model(features):
    return torch.matmul(features, w.T) + b

# 定义损失函数
def loss(y_hat, y):
    return (y_hat - y) ** 2 / 2

# 定义优化器
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 训练
def train(features, labels, num_epochs, batch_size, lr=0.01):
    # 训练
    for epoch in range(num_epochs):
        for X, y in get_batch(features, labels, batch_size=batch_size):
            y_hat = model(X)
            l = loss(y_hat, y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        # 每轮进行一次评估
        with torch.no_grad():
            l = loss(model(features), labels)
            print(f'epoch: {epoch + 1} | train_loss: {l.mean()}')

if __name__ == '__main__':
    num_epochs = 3
    batch_size = 20
    train(features, labels, num_epochs, batch_size, lr=0.02)
