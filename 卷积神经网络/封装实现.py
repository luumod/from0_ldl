from torch import nn
import torchvision
from torchvision import transforms
from torch.utils import data
from typing import Literal
import torch
from torch import Tensor
from torch import device
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 每张图片为 28×28
            # 特征提取器 = 卷积层 + 激活函数 + 池化层
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), nn.Sigmoid(),  # (BatchSize, 6, 28, 28)
            nn.AvgPool2d(kernel_size=2, stride=2),  # (BatchSize, 6, 14, 14)

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), nn.Sigmoid(),  # (BatchSize, 16, 10, 10)
            nn.AvgPool2d(kernel_size=2, stride=2),  # (BatchSize, 16, 5, 5)

            # 分类器（展平后依次进入三个全连接层）
            nn.Flatten(),  # (BatchSize, 16×5×5)
            nn.Linear(in_features=16 * 5 * 5, out_features=120), nn.Sigmoid(),  # (BatchSize, 120)
            nn.Linear(in_features=120, out_features=84), nn.Sigmoid(),  # (BatchSize, 84)
            nn.Linear(in_features=84, out_features=10)  # (BatchSize, 10)
        )

    def forward(self, x) -> Tensor:
        return self.model(x)


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader[list[Tensor, Tensor]],
                 test_loader: DataLoader[list[Tensor, Tensor]],
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 platform: device,
                 log_dir: str = './runs'):
        self.model: nn.Module = model.to(platform)
        self.train_loader: DataLoader[list[Tensor, Tensor]] = train_loader
        self.test_loader: DataLoader[list[Tensor, Tensor]] = test_loader
        self.criterion: nn.Module = criterion
        self.optimizer: optim.Optimizer = optimizer
        self.platform: device = platform
        self.writer = SummaryWriter(log_dir)

    def _run_epoch(self, data_loader: DataLoader[list[Tensor, Tensor]], mode: Literal['train', 'eval']):
        assert mode in ['train', 'eval'], "mode must be either 'train' or 'eval'"
        total_loss = correct_count = total_count = 0.0
        is_train: bool = mode == 'train'

        self.model.train() if is_train else self.model.eval()

        with torch.set_grad_enabled(is_train):
            for features, labels in data_loader:
                features: Tensor = features.to(self.platform)
                labels: Tensor = labels.to(self.platform)

                outputs: Tensor = self.model(features)  # 前向传播
                loss: Tensor = self.criterion(outputs, labels) # 计算损失

                if is_train:  # 反向传播与优化（只有在训练时）
                    self.optimizer.zero_grad()
                    loss.backward()       # 反向传播
                    self.optimizer.step() # 参数更新

                total_loss += loss.item() # 记录总训练损失

                _, predicted = outputs.max(dim=1)
                total_count += labels.size(0)
                correct_count += (predicted == labels).sum().item() # 记录总的正确数量

        accuracy = 100 * correct_count / total_count
        loss = total_loss / len(data_loader)
        return accuracy, loss

    def train(self, epochs_num: int):
        for epoch in range(epochs_num):
            accuracy_train, loss_train = self._run_epoch(self.train_loader, mode='train')
            accuracy_test, loss_test = self.evaluate()  # 调用评估方法

            # TODO: 记录训练损失和精度。可以考虑使用装饰器输出或 TensorBoard 数据
            self.writer.add_scalar('Accuracy/train', accuracy_train, global_step=epoch + 1)
            self.writer.add_scalar('Loss/train', loss_train, global_step=epoch + 1)
            self.writer.add_scalar('Accuracy/test', accuracy_test, global_step=epoch + 1)
            self.writer.add_scalar('Loss/test', loss_test, global_step=epoch + 1)

            print(f'第 {epoch + 1:03}/{epochs_num} 轮，'
                  f'训练损失：{loss_train:.4f}，训练精度：{accuracy_train:05.2f}%，'
                  f'测试损失：{loss_test:.4f}，测试精度：{accuracy_test:05.2f}%')

    def evaluate(self):
        accuracy_test, loss_test = self._run_epoch(self.test_loader, mode='eval')
        return accuracy_test, loss_test

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()


if __name__ == '__main__':
    BATCH_SIZE = 256
    EPOCHS_NUM = 10
    LEARNING_RATE = 0.9

    model = LeNet()
    train_loader, test_loader = load_data_fashion_mnist(BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), LEARNING_RATE)
    platform = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with Trainer(model, train_loader, test_loader, criterion, optimizer, platform) as trainer:
        trainer.train(EPOCHS_NUM)
        trainer.evaluate()

