import torch
from torch import nn

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
x = torch.rand((2, 4))
print(model(x))

print(model[2].state_dict())
print(type(model[2].bias))
print(model[2].bias)
print(model[2].bias.data)
print(model[2].weight.grad == None)

print(*[(name, param.shape) for name, param in model[0].named_parameters()])
print(*[(name, param.shape) for name, param in model.named_parameters()])

print(model.state_dict()['2.bias'].data)

print('-------------------------------------')

def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    model = nn.Sequential()
    for i in range(4):
        model.add_module(f'block {i}', block1())
    return model

model = nn.Sequential(block2(), nn.Linear(4,1))
print(model(x))
print(model)
print(model[0][1][0].bias.data)
print(model[0][1][0].weight.data)


# 内置初始化
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
model.apply(init_normal)
print(model[0].weight.data)
print(model[0].bias.data) # all 0

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
model.apply(init_constant)
print(model[0].weight.data)
print(model[0].bias.data)

def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

model[0].apply(init_xavier)
model[2].apply(init_42)
print(model[2].weight.data)