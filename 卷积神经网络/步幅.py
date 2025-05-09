import torch
import torch.nn as nn

def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)

X = torch.randn((8, 8)) # 8*8可以被2整除，所以相当于减半
print(comp_conv2d(conv2d, X).shape)
