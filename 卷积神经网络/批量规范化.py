import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils import data
from torch.nn import BatchNorm1d, BatchNorm2d

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize is not None:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST('./data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST('./data', train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size=batch_size,shuffle=True),
            data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False))


# 自定义实现
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        # 预测模式
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 全连接层
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 二维卷积层
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
        Y = gamma * X_hat + beta
        return Y, moving_mean.data, moving_var.data

class LeNetModel(nn.Module):
    def __init__(self):
        super(LeNetModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), # 6*28*28
            nn.BatchNorm2d(6),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2), # 6*14*14

            nn.Conv2d(6, 16, kernel_size=5), # 16*10*10
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2), # 16*5*5

            nn.Flatten(),

            nn.Linear(16*5*5, 120),
            nn.BatchNorm1d(120),
            nn.Sigmoid(),

            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.Sigmoid(),

            nn.Linear(84, 10)
        )

    def forward(self, X):
        return self.model(X)


class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def run_epoch(self, data_loader, mode):
        assert mode in ['train', 'eval'], 'mode must be either "train" or "eval".'
        # 总损失 | 正确数量 | 总数量
        total_loss = correct_count = total_count = 0.0
        is_train = mode == 'train'
        self.model.train() if is_train else self.model.eval()

        with torch.set_grad_enabled(is_train):
            for features, labels in data_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                # 前向传播
                outputs = self.model(features)
                # 计算损失
                loss = self.criterion(outputs, labels)
                if is_train:
                    # 梯度清零
                    self.optimizer.zero_grad()
                    # 反向传播
                    loss.backward()
                    # 参数更新
                    self.optimizer.step()
                total_loss += loss.item()

                _, predicted =  outputs.max(dim=1)
                total_count += labels.shape[0]
                correct_count += (predicted == labels).sum().item()
        accuracy = 100 * correct_count / total_count
        loss = total_loss / len(data_loader)
        return accuracy, loss

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_accuracy, train_loss = self.run_epoch(self.train_loader, mode='train')
            test_accuracy, test_loss = self.evaluate()
            print(f'第 {epoch + 1:03}/{num_epochs} 轮，'
                  f'训练损失：{train_loss:.4f}，训练精度：{train_accuracy:05.2f}%，'
                  f'测试损失：{test_loss:.4f}，测试精度：{test_accuracy:05.2f}%')

    def evaluate(self):
        test_accuracy, test_loss = self.run_epoch(self.test_loader, mode='eval')
        return test_accuracy, test_loss

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def save_model(self):
        torch.save(self.model.state_dict(), './models/googleNet.pt')
        print('模型保存成功')

if __name__ == '__main__':
    batch_size = 256
    num_epochs = 12
    lr = 0.9

    model = LeNetModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_loader, test_loader = load_data_fashion_mnist(batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with Trainer(model, train_loader, test_loader, criterion, optimizer, device) as trainer:
        trainer.train(num_epochs)



