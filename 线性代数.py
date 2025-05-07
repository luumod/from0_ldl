import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

print( x ** y)

x = torch.arange(4)
print(x)
print(x.shape)
print(x.size())

x = torch.arange(20).reshape(2,10)
print(x)
print(x.T)
print(x.t())
print(x.transpose(0,1).transpose(0,1))

x = torch.arange(24).reshape(2,3,4)
print(x)
y = x.clone()
yy = x + y
print(x == y)
print(yy)
print(x * y) # 对应元素相乘
a = 2
print(x + 2)
print((x * 2).shape)

x = torch.arange(24,dtype=torch.float32).view(3,-1,2)
print(x)
print(x.sum())
print(x.sum(dim=0))
print(x.sum(dim=1))
print(x.sum(dim=2))

x = torch.arange(8, dtype=torch.float32).reshape(2,4)
print(x)
print(x.sum())
print(x.sum(dim=0))
print(x.sum(dim=1))
print(x.sum(dim=[0,1]))
print(x.numel())
print(x.sum() / x.numel())
print(x.mean(dim=0))
print(x.mean(dim=1))
print(x.sum(dim=0)) # 降维
print(x.sum(dim=0,keepdim=True)) # 非降维 1*4
y = x.sum(dim=0,keepdim=True)
print( x / y)
print(x.cumsum(dim=0)) # 累加且非降维
print(x.cumsum(dim=1))

x = torch.arange(12,dtype=torch.float32).reshape(3,4)
y = torch.arange(12,dtype=torch.float32).reshape(4,3)
print(torch.mm(x,y))

x = torch.arange(6,dtype=torch.float32).reshape(2,3)
y = torch.arange(6,dtype=torch.float32).reshape(2,3)
print(x.T + y.T == (x + y).T)

x = torch.arange(24,dtype=torch.float32).reshape(3,8)
print(x)
print(x.sum(dim=1,keepdim=True))
print(x / x.sum(dim=1,keepdim=True))
