import time
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
from torch.utils import data

model = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), # 6*28*28
    nn.Sigmoid(),

    nn.AvgPool2d(kernel_size=2, stride=2), # 6*14*14

    nn.Conv2d(6, 16, kernel_size=5), # 16*10*10
    nn.Sigmoid(),

    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Flatten(), # 展平

    nn.Linear(16*5*5, 120),
    nn.Sigmoid(),

    nn.Linear(120, 84),
    nn.Sigmoid(),

    nn.Linear(84, 10),
)

# 打印模型
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in model:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)


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

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum()) # 返回预测成功的个数

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

# 评估模式
def evaluate_accuracy(model, data_iter, device=None):
    if isinstance(model, nn.Module):
        model.eval()
        if not device:
            device = next(iter(model.parameters())).device()
    num_correct, num_all = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X] # 取出每一个值放入到GPU中
            else:
                X = X.to(device)
            y = y.to(device)
            num_correct = accuracy(model(X), y)
            num_all = y.numel()
    return num_correct / num_all

def train(model, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    print('train on ', device)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer = Timer()
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        model.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                # 训练总损失 | 训练准确率 | 训练样本总数
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]   # 平均训练每个样本的损失
            train_acc = metric[1] / metric[2] # 平均训练每个样本的准确率
        test_acc = evaluate_accuracy(model, test_iter, device)
    print(f'loss: {train_l:.3f} | train_acc: {train_acc:.3f} | test_acc: {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {device}')


lr, num_epochs = 0.9, 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train(model, train_iter, test_iter, num_epochs, lr, device)

