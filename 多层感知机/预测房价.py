import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from utils import load_array

# -----------------读取数据--------------------
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')
print(train_data.shape) # 1460*81(80个特征，1个标签)
print(test_data.shape)  # 1459*80

# -----------------数据清洗--------------------
# 去掉id列（无用列），并且将训练与测试集结合起来统一进行数据的清洗
all_features = pd.concat((train_data.iloc[:,1:-1], test_data.iloc[:, 1:]))
# 连续值处理：减去均值，除以方差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply( # 对列操作
    lambda x: (x - x.mean()) / (x.std())
)
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# 离散值的处理：独热编码（注意显式类型为float32）
all_features = pd.get_dummies(all_features, dummy_na=True, dtype=np.float32)
print(all_features.shape)

# -----------------获取数据集--------------------
n_train = train_data.shape[0] # 训练样本个数
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1,1), dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)

print(train_features.shape)
print(test_features.shape)
print(train_labels.shape)
# torch.Size([1460, 330])
# torch.Size([1459, 330])
# torch.Size([1460, 1])


# 损失函数：均方误差
loss = nn.MSELoss()
# 输入特征数
in_features = train_features.shape[1]

# 定义模型：简单线性模型
def get_model():
    model = nn.Sequential(nn.Linear(in_features, 1))
    return model

# 对数均方根误差
def log_rmse(model, features, labels):
    # 带log的均方误差
    clipped_preds = torch.clamp(model(features), 1,float('inf'))
    rmse = torch.sqrt(loss(
        torch.log(clipped_preds),
        torch.log(labels)
    ))
    return rmse.item()

# 训练(训练集 + 验证集) 或者 (全部作为训练集)
def train(model, train_features, train_labels, test_features, test_labels,
          num_epochs, lr, weight_decay, batch_size):
    train_loss, test_loss = [], []
    # 训练集
    train_iter = load_array((train_features, train_labels), batch_size)
    # Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            y_hat = model(X)
            # 计算误差
            l = loss(y_hat, y)
            # 反向传播
            l.backward()
            # 更新参数
            optimizer.step()
        train_loss.append(log_rmse(model, train_features, train_labels))
        if test_labels is not None:
            test_loss.append(log_rmse(model ,test_features, test_labels))
    return train_loss, test_loss

# K折交叉验证
def get_k_fold_data(k, i, X, y):
    '''
    :param k: 将训练集分成k个块
    :param i: 选择第i个块作为验证数据集
    :param X: 训练集的特征
    :param y: 训练集的标签
    :return:
    '''
    assert k > 1
    fold_size = X.shape[0] // k  # 365
    X_train, y_train = None, None
    for j in range(k): # 0 1 2 3
        idx = slice(j * fold_size, (j + 1) * fold_size)  # [0, 365) [365, 700) ...
        X_part, y_part = X[idx, :], y[idx] # 365*80  365
        if j == i:
            # 第i块作为验证集
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0) # 1460*80
            y_train = torch.cat([y_train, y_part], 0) # k*365
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, lr, weight_decay, batch_size):
    train_loss_sum, valid_loss_sum = 0, 0
    for i in range(k):
        # 选第i块作为验证集
        data = get_k_fold_data(k, i, X_train, y_train)
        model = get_model()
        train_loss, valid_loss = train(model, *data, num_epochs=num_epochs,
                                       lr=lr, weight_decay=weight_decay,batch_size=batch_size)
        train_loss_sum += train_loss[-1]
        valid_loss_sum += valid_loss[-1]
        print(f'折{i+1}，训练log_rmse: {float(train_loss[-1])}, 验证log_rmse: {float(valid_loss[-1])}')
    return train_loss_sum / k, valid_loss_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-折验证：平均训练log_rmse: {float(train_l)}, 平均验证log_rmse: {float(valid_l)}')


# 全部数据集作为训练集训练后预测
def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    model = get_model()
    train_loss, _ = train(model, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    print(f'训练log_rmse: {float(train_loss[-1])}')
    # 应用于测试集
    preds = model(test_features).detach().numpy()
    # 导出数据到csv中
    test_data['SalePrice'] = pd.Series(preds.reshape(1,-1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./data/submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)

