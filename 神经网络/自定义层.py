import torch
import torch.nn as nn
import torch.nn.functional as F

class GenteredLayer(nn.Module):
    def __init__(self):
        super(GenteredLayer, self).__init__()

    def forward(self, X):
        return X - X.mean()

layer = GenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

model = nn.Sequential(nn.Linear(8, 128), GenteredLayer())
x = model(torch.randn(4, 8))
print(x)

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super(MyLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
print(linear.weight)
print(linear.bias)

print(linear(torch.randn(2, 5)))

model = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(model(torch.rand(2, 64)))
