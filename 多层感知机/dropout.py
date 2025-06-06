import torch
import torch.nn as nn
from utils import load_data_fashion_mnist, train_ch3

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

# 两个隐藏层的多层感知机
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 28*28, 10, 256, 256

dropout1, dropout2 = 0.2, 0.5
class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(Model, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        if self.training == True:
            # 只有在训练时使用dropout
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

model = Model(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

# 训练测试
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = load_data_fashion_mnist(batch_size)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# train_ch3(model, train_iter, test_iter, loss, num_epochs, optimizer)


# 简洁实现
model = nn.Sequential(nn.Flatten(),

                      nn.Linear(28*28, 256),
                      nn.ReLU(),
                      nn.Dropout(dropout1),

                      nn.Linear(256, 256),
                      nn.ReLU(),
                      nn.Dropout(dropout2),

                      nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

model.apply(init_weights)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
train_ch3(model, train_iter, test_iter, loss, num_epochs, optimizer)