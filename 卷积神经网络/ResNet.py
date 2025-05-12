import torch
import torch.nn as nn
from torchinfo import summary
from torchvision import transforms
from torch.utils import data
import torchvision
import torch.nn.functional as F

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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_bottleneck=False):
        super(ResidualBlock, self).__init__()
        if use_bottleneck:
            mid_channels = out_channels // 4
            self.main_path = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(),

                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(),

                nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.main_path = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),

                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # 匹配维度用于跳跃连接
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity() # 恒等映射
    def forward(self, X):
        out = self.main_path(X) + self.shortcut(X) # f(x) = x + R(x)
        out = F.relu(out) # 激活函数
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 首先是两个连续的，输入输出通道数均为64的残差快
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),

            # 三个残差快
            ResidualBlock(64, 128, stride=2), # 将尺寸减半，通道数翻倍
            ResidualBlock(128, 128, use_bottleneck=True), # 输出和输出通道数一致，但是使用瓶颈结构减少计算量

            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, use_bottleneck=True),

            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, use_bottleneck=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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



# from torchinfo import summary
#
# model = ResNet(num_classes=10)
# summary(model, input_size=(1, 1, 224, 224))

if __name__ == '__main__':
    BATCH_SIZE = 256
    EPOCHS_NUM = 30
    LEARNING_RATE = 0.01

    model = ResNet(num_classes=10)
    train_loader, test_loader = load_data_fashion_mnist(BATCH_SIZE, resize=96)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE)
    platform = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with Trainer(model, train_loader, test_loader, criterion, optimizer, platform) as trainer:
        trainer.train(EPOCHS_NUM)