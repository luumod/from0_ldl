import torch
import torch.nn as nn
import torch.nn.functional as F

# 一个256单元和一个激活函数的隐藏层 和 一个10个隐藏单元无隐藏层的输出层
model = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 1))
x = torch.randn(30, 20)
print(model(x)) # 30*1

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(20, 256) # 隐藏层
        self.output = nn.Linear(256, 1)  # 输出层

    def forward(self, X):
        return self.output(F.relu(self.hidden(X)))

model = MLP()
print(model(x))

class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X

model = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 128),nn.ReLU(),nn.Linear(128,1))
print(model(x))

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super(FixedHiddenMLP, self).__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        return X.sum()

model = FixedHiddenMLP()
print(model(x))

class NestMLP(nn.Module):
    def __init__(self):
        super(NestMLP, self).__init__()
        self.lin1 = nn.Sequential(nn.Linear(20, 256),nn.ReLU(),
                                 nn.Linear(256, 128),nn.ReLU())
        self.lin2 = nn.Linear(128, 64)

    def forward(self, X):
        return self.lin2(self.lin1(X))

model = nn.Sequential(NestMLP(), nn.ReLU(), nn.Linear(64, 20), FixedHiddenMLP())
print(model(x))