import torch
import torchvision
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from utils import train_ch3

def load_data_fashion_mnist(batch_size):
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size=batch_size,shuffle=True),
            data.DataLoader(mnist_test, batch_size=batch_size,shuffle=False))

batch_size = 256

train_iter, test_iter =load_data_fashion_mnist(batch_size=batch_size)

# 初始化模型参数
num_inputs, num_outputs, num_hiddens = 28*28, 10, 256
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens,requires_grad=True) * 0.01)
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs,requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1,b1,W2,b2]

# 定义relu激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# 定义模型
def model(X):
    X = X.reshape((-1, num_inputs)) # 256*784
    H = relu(X @ W1 + b1) # 256*256
    return H @ W2 + b2 # 256*10

# 损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 训练
num_epochs = 10
lr = 0.01
optimizer = torch.optim.SGD(params, lr=lr)
train_ch3(model, train_iter, test_iter, loss, num_epochs, optimizer)


