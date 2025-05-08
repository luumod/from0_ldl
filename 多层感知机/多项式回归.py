import math
import numpy as np
import torch
import torch.nn as nn
from utils import Accumulator, train_epoch_ch3, load_array

max_degree = 20
n_train, n_test = 100, 100
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6]) # 20,

features = np.random.normal(size=(n_train + n_test, 1)) # 200*1
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1)) # 200*20
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)
labels = np.dot(poly_features, true_w) # 200*20 * 20*1(自动扩展)
labels += np.random.normal(scale=0.1, size=labels.shape)

true_w, features, poly_features, labels = [
    torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]
]

print(features[:2])
print(poly_features[:2, :])
print(labels.shape)

def evaluate_loss(model, data_iter, loss):
    metric = Accumulator(2)
    for X, y in data_iter:
        out = model(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

# 训练
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    model = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = load_array((train_features, train_labels.reshape(-1, 1)), batch_size,is_train=True)
    test_iter = load_array((test_features, test_labels.reshape(-1,1)), batch_size,is_train=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        train_epoch_ch3(model, train_iter, loss, optimizer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(f'{evaluate_loss(model, train_iter, loss)}')
            print(f'{evaluate_loss(model, test_iter, loss)}')
    print('weight:',model[0].weight.data.numpy())

# 正常
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])

# 欠拟合
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])

# 过拟合
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:])