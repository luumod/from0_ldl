# import torch
import torchvision
from torch.utils import data
from torchvision import transforms
# from timer import Timer
#
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True),
            data.DataLoader(mnist_test, batch_size, shuffle=False))

# batch_size = 256
# train_iter, test_iter = load_data_fashion_mnist(batch_size)
#
# num_inputs = 28 * 28
# num_outputs = 10
#
# W = torch.normal(0, 0.01, (num_inputs, num_outputs), requires_grad=True)
# b = torch.zeros(num_outputs,requires_grad=True)
#
# def softmax(X):
#     X_exp = torch.exp(X)
#     partition = X_exp.sum(dim=1, keepdim=True)
#     return X_exp / partition
#
# # 小批量样本的矢量化
# def model(X):
#     # X:  batch_size(批量大小) * d(特征维度)
#     # W:  d(特征维度) * c(类别个数)
#     # y = XW + b ---> batch_size(批量大小) * c(类别个数)
#     return softmax(torch.matmul(X.reshape((-1,W.shape[0])), W) + b)
#
# # y = torch.tensor([0, 2, 1])
# # y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5], [0.2, 0.6, 0.2]])
#
# def cross_entropy(y_hat, y):
#     return -torch.log(y_hat[range(len(y_hat)), y])
#
# # print(cross_entropy(y_hat, y))
#
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum()) # 返回预测成功的个数

# print(accuracy(y_hat, y) / len(y))

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(model, data_iter):
    if isinstance(model ,torch.nn.Module):
        model.eval()
    metric = Accumulator(2) # 正确数量 | 总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(model(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(model, train_iter, loss, updater):
    if isinstance(model, torch.nn.Module):
        model.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = model(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(model, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(model, train_iter, loss, updater)
        test_acc = evaluate_accuracy(model, test_iter)
        train_loss, train_acc = train_metrics
        print(f'train_acc: {train_acc} | train_loss: {train_loss} | test_acc: {test_acc}')

# lr = 0.1
#
# def sgd(params, lr, batch_size):
#     with torch.no_grad():
#         for param in params:
#             param -= lr * param.grad / batch_size
#             param.grad.zero_()
#
# def updater(batch_size):
#     return sgd([W, b], lr=lr, batch_size=batch_size)
#
# num_epochs = 10
# train_ch3(model, train_iter, test_iter, cross_entropy, num_epochs, updater)
#
#


# ------------------------------------------
import torch
import torch.nn as nn

batch_size  = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

model = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

model.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

num_epochs = 10

train_ch3(model, train_iter, test_iter, loss, num_epochs, optimizer)

