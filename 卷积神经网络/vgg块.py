import torch.nn as nn
import torch
from torchinfo import summary
from torchvision import transforms
from torch.utils import data
import torchvision

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

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, X):
        return self.block(X)

class VGG11(nn.Module):
    def __init__(self, num_classes):
        super(VGG11, self).__init__()
        self.extractor = nn.Sequential(
            VGGBlock(1, 16),
            VGGBlock(16, 32),
            VGGBlock(32, 64),
            VGGBlock(64, 128),
            VGGBlock(128, 128)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(4096, num_classes)
        )

    def forward(self, X):
        X = self.extractor(X)
        X = torch.flatten(X, 1)
        X = self.classifier(X)
        return X

if __name__ == '__main__':
    batch_size = 128
    num_epochs = 30
    lr = 0.05

    model = VGG11(num_classes=10)
    train_loader, test_loader = load_data_fashion_mnist(batch_size, resize=224)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with Trainer(model, train_loader, test_loader, criterion, optimizer, device) as trainer:
        trainer.train(num_epochs)

