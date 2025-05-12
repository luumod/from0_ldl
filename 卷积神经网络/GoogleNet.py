import torch
import torch.nn as nn
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

    def save_model(self):
        torch.save(self.model.state_dict(), './models/googleNet.pt')
        print('模型保存成功')

class IncepBlock(nn.Module):
    def __init__(self, in_channels, c1_out, c2_out, c3_out, c4_out):
        super(IncepBlock, self).__init__()
        self.channel1 = nn.Sequential(
            nn.Conv2d(in_channels, c1_out, kernel_size=1),
            nn.ReLU()
        )
        self.channel2 = nn.Sequential(
            nn.Conv2d(in_channels, c2_out[0], kernel_size=1),# 96*12*12
            nn.ReLU(),
            nn.Conv2d(c2_out[0], c2_out[1], kernel_size=3, padding=1), # 128*12*12
            nn.ReLU()
        )
        self.channel3 = nn.Sequential(
            nn.Conv2d(in_channels, c3_out[0], kernel_size=1), # 16*12*12
            nn.ReLU(),
            nn.Conv2d(c3_out[0], c3_out[1], kernel_size=5, padding=2), #32*12*12
            nn.ReLU()
        )
        self.channel4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1), # 192*12*12
            nn.Conv2d(in_channels,c4_out, kernel_size=1),  #32*12*12
            nn.ReLU()
        )
    def forward(self, X):
        o1 = self.channel1(X)
        o2 = self.channel2(X)
        o3 = self.channel3(X)
        o4 = self.channel4(X)
        output = torch.cat([o1, o2, o3, o4], dim=1)
        return output

class GoogleNet(nn.Module):
    def __init__(self, num_classes):
        super(GoogleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), # 64*48*48
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),     # 64*24*24

            nn.Conv2d(64, 64, kernel_size=1), # 64*24*24
            nn.ReLU(),

            nn.Conv2d(64, 192, kernel_size=3, padding=1), #192*24*24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), #192*12*12

            IncepBlock(in_channels=192, c1_out=64, c2_out=(96, 128), c3_out=(16, 32), c4_out=32), #
            IncepBlock(in_channels=256, c1_out=128, c2_out=(128, 192), c3_out=(32, 96), c4_out=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            IncepBlock(in_channels=480, c1_out=192, c2_out=(96, 208), c3_out=(16, 48), c4_out=64),
            IncepBlock(in_channels=512, c1_out=160, c2_out=(112, 224), c3_out=(24, 64), c4_out=64),
            IncepBlock(in_channels=512, c1_out=128, c2_out=(128, 256), c3_out=(24, 64), c4_out=64),
            IncepBlock(in_channels=512, c1_out=112, c2_out=(144, 288), c3_out=(32, 64), c4_out=64),
            IncepBlock(in_channels=528, c1_out=256, c2_out=(160, 320), c3_out=(32, 128), c4_out=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            IncepBlock(in_channels=832, c1_out=256, c2_out=(160, 320), c3_out=(32, 128), c4_out=128),
            IncepBlock(in_channels=832, c1_out=384, c2_out=(192, 384), c3_out=(48, 128), c4_out=128),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_features=1024, out_features=num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    BATCH_SIZE = 128
    EPOCHS_NUM = 30
    LEARNING_RATE = 0.005

    model = GoogleNet(num_classes=10)
    train_loader, test_loader = load_data_fashion_mnist(BATCH_SIZE, resize=96)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE)
    platform = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with Trainer(model, train_loader, test_loader, criterion, optimizer, platform) as trainer:
        trainer.train(EPOCHS_NUM)
        trainer.save_model()