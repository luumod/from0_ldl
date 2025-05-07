import torch

x = torch.arange(28*28*256).reshape(256,1,28,28)
W = torch.arange(784*10).reshape(784,10)
y = x.reshape((-1,W.shape[0])) # 256 * 784
print(x.shape)
print(y.shape)
print(W.shape)

yy = torch.matmul(y, W) # 256*784 * 784*10
print(yy.shape)