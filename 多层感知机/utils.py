import torch
from torch.utils import data
import torchvision
from torchvision import transforms

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
            # 使用内置优化器
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 自定义优化器
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

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

def synthetic_data(w, b, num_examples):
    # X是输入：样本数*特征数
    X = torch.normal(0, 1, (num_examples, len(w)))
    # y是输出：样本数*输出维度
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def linreg(X, w, b):
    return torch.matmul(X, w) + b

def square_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

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
