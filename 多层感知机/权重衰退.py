import torch
import torch.nn as nn
from utils import synthetic_data, load_array, linreg, square_loss, sgd, evaluate_accuracy

# 20个训练样本
n_train = 20
# 100个测试样本
n_test = 100
# 每个样本有200个特征
num_inputs = 200
# 批量大小
batch_size = 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = synthetic_data(true_w, true_b, n_train)
train_iter = load_array(train_data, batch_size)
test_data = synthetic_data(true_w, true_b, n_test)
test_iter = load_array(test_data, batch_size, is_train=False)

def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

# 训练
def train(lambd):
    w, b = init_params()
    model = lambda X: linreg(X, w, b,)
    loss = square_loss
    num_epochs, lr = 100, 0.03
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(model(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
    print('w的L2范数是：', torch.norm(w).item())

# 忽略正则化
train(lambd=0)

# 使用权重衰退
train(lambd=3)

# 使用内置库
def train_concise(wd):
    model = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in model.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置bias无衰退
    optimizer = torch.optim.SGD([
        {'params':model[0].weight, 'weight_decay': wd},
        {'params':model[0].bias}
    ], lr=lr)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(model(X), y)
            l.mean().backward()
            optimizer.step()
    print('w的L2范数是：', model[0].weight.norm().item())

train_concise(0)

train_concise(3)