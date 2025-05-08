import torch
import matplotlib.pyplot as plt

# relu
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)

y.backward(torch.ones_like(x), retain_graph=True)

# sigmoid
y = torch.sigmoid(x)

# tanh
y = torch.tanh(x)
plt.plot(x.detach(), y.detach(), 'x', 'tanh(x)')
plt.show()

