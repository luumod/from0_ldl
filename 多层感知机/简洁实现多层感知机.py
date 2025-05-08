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

# 模型初始化
model = nn.Sequential(nn.Flatten(),

                      # 与softmax的区别就是添加了隐藏层(线性层 + relu激活函数)
                      nn.Linear(28*28, 256),
                      nn.ReLU(),

                      nn.Linear(256, 10))

# 对weights采用自定义初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

model.apply(init_weights)

# 训练
num_epochs = 10
lr = 0.01
batch_size = 256
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
train_ch3(model, train_iter, test_iter, loss, num_epochs, optimizer)



