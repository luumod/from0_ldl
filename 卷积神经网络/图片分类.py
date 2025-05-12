import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils import data

class LeafDataset(Dataset):
    def __init__(self, data_root, csv_path, transform=None, mode='train'):
        """
        Args:
            data_root (string): 数据根目录路径 (包含images文件夹的目录)
            csv_path (string): CSV文件路径
            transform (callable, optional): 图像预处理转换
            mode (str): 'train' 或 'test' 模式
        """
        self.data_root = data_root
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.mode = mode

        # 训练模式需要处理标签
        if self.mode == 'train':
            self.image_paths = self.df.iloc[:1000, 0].values
            self.labels = self.df.iloc[:1000, 1].values
            self.label_encoder = LabelEncoder()
            self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        else:
            self.image_paths = self.df.iloc[:1000, 0].values

    def __len__(self):
        # return len(self.df)
        return 1000

    def __getitem__(self, idx):
        # 拼接完整图片路径
        img_path = os.path.join(self.data_root, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')  # 确保转换为RGB格式

        if self.transform:
            image = self.transform(image)

        if self.mode == 'train':
            label = self.encoded_labels[idx]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            return image  # 测试模式只返回图像

def create_loaders(data_root='./data/classify-leaves',
                   batch_size=32,
                   train_csv='train.csv',
                   test_csv='test.csv',
                   image_size=224,
                   num_workers=4):
    """
    创建训练和测试DataLoader

    Args:
        data_root: 数据根目录
        batch_size: 批大小
        train_csv: 训练集CSV文件名
        test_csv: 测试集CSV文件名
        image_size: 输入图像尺寸
        num_workers: 数据加载线程数

    Returns:
        train_loader, valid_loader, test_loader
    """
    # 数据增强和归一化配置
    train_transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.RandomRotation(15),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(image_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = LeafDataset(
        data_root=data_root,
        csv_path=os.path.join(data_root, train_csv),
        transform=train_transform,
        mode='train'
    ) # images | labels

    test_dataset = LeafDataset(
        data_root=data_root,
        csv_path=os.path.join(data_root, test_csv),
        transform=test_transform,
        mode='test'
    ) # images

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_datasets, valid_datasets = random_split(train_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader

class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, criterion, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def run_epoch(self, data_loader, mode):
        assert mode in ['train', 'eval'], 'mode must be either "train" or "eval"'
        total_loss = correct_count = total_count = 0.0
        is_train = mode == 'train'
        self.model.train() if is_train else self.model().eval()

        with torch.set_grad_enabled(is_train):
            for features, labels in tqdm(data_loader):
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()
                _, predicted = outputs.max(dim=1)
                total_count += labels.shape[0]
                correct_count += (predicted == labels).sum().item()
        accuracy = 100 * correct_count / total_count
        loss = total_loss / len(data_loader)
        return accuracy, loss

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_accuracy, train_loss = self.run_epoch(self.train_loader, mode='train')
            # valid_accuracy, valid_loss = self.valid()

            print(f'第 {epoch + 1:03}/{num_epochs} 轮，'
                  f'训练损失：{train_loss:.4f}，训练精度：{train_accuracy:05.2f}%，')
                  #f'验证损失：{valid_loss:.4f}，验证精度：{valid_accuracy:05.2f}%')

    def valid(self):
        test_accuracy, test_loss = self.run_epoch(self.valid_loader, mode='eval')
        return test_accuracy, test_loss

    def evaluate(self):
        test_accuracy, test_loss = self.run_epoch(self.test_loader, mode='eval')
        return test_accuracy, test_loss

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


if __name__ == "__main__":
    batch_size = 64
    lr = 0.01
    num_epochs = 20

    train_loader, valid_loader, test_loader = create_loaders(batch_size=batch_size)


    # 演示如何读取一个批次数据
    # 训练数据维度: torch.Size([64, 1, 224, 224])
    # 标签维度: torch.Size([64])

    model = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with Trainer(model, train_loader, valid_loader, test_loader, criterion, optimizer, device) as trainer:
        trainer.train(num_epochs=num_epochs)
        trainer.evaluate()
    # trans = [transforms.ToTensor()]
    # trans = transforms.Compose(trans)
    # mnist_tv = torchvision.datasets.FashionMNIST(
    #     root="./data", train=True, transform=trans, download=True)
    # mnist_test = torchvision.datasets.FashionMNIST(
    #     root="./data", train=False, transform=trans, download=True)
    #
    # train_size = int(0.8 * len(mnist_tv))
    # valid_size = len(mnist_tv) - train_size
    # train_mnist, valid_mnist = random_split(mnist_tv, [train_size, valid_size])
    #
    # train_loader = DataLoader(train_mnist, batch_size=256, shuffle=True)
    # valid_loader = DataLoader(valid_mnist, batch_size=256, shuffle=True)
    # test_loader = DataLoader(mnist_test, batch_size=256, shuffle=False)
    #
    # print('加载完毕')


