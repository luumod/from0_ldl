import torch
import torch.nn as nn

def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1) # p_all = k - 1
X = torch.randn(size=(8, 8))
y = comp_conv2d(conv2d, X)
print(y)

conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1)) # p_h_all = k_h -1, p_w_all = k_w - 1
y = comp_conv2d(conv2d, X)
print(y.shape)

