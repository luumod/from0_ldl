import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision import transforms
from torchinfo import summary


def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True),
            data.DataLoader(mnist_test, batch_size, shuffle=False))


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model = nn.Sequential(
                    # 1*224*224的输入
                    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), # 96*54*54
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2), # 96*26*26

                    nn.Conv2d(96, 256, kernel_size=5, padding=2), # 256*26*26
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2), # 256*12*12

                    nn.Conv2d(256, 384, kernel_size=3, padding=1), # 384*12*12
                    nn.ReLU(),
                    nn.Conv2d(384, 384, kernel_size=3, padding=1), # 384*12*12
                    nn.ReLU(),
                    nn.Conv2d(384, 256, kernel_size=3, padding=1), # 256*12*12
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2),         # 256*5*5

                    nn.Flatten(),

                    nn.Linear(256*5*5, 4096),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),

                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),

                    nn.Linear(4096, 10),
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
                # features: [128, 1, 224, 224]
                # labels:   [128, 1]
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

if __name__ == '__main__':
    batch_size = 128
    lr = 0.01
    num_epochs = 30

    model = AlexNet()
    train_loader, test_loader = load_data_fashion_mnist(batch_size, resize=224)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with Trainer(model, train_loader, test_loader, criterion, optimizer, device) as trainer:
        trainer.train(num_epochs)