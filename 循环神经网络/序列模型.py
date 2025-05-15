import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# sample_size = 1000
# tau = 4
# time_steps = torch.arange(1, sample_size + tau + 1, dtype=torch.float32)
# values = torch.sin(time_steps * 0.01) + torch.normal(0, 0.2, size=(sample_size + tau,))
#
# features = torch.stack([values[i:i+tau] for i in range(sample_size)], dim=0)
# labels = values[tau:]

def generate_data_from_sin(mean, std, size, tau):
    time_steps = torch.arange(1, size + tau +1, dtype=torch.float32)
    values = torch.sin(time_steps * 0.01) + torch.normal(mean, std,size=(size + tau, ))
    return time_steps, values

def generate_features_and_labels(values, size, tau):
    features = torch.stack([values[i:i+tau] for i in range(size)], dim=0)
    labels = values[tau:]
    return features, labels

def get_dataloader(features, labels, train_size, batch_size):
    train_dataset = TensorDataset(features[:train_size], labels[:train_size])
    valid_dataset = TensorDataset(features[train_size:], labels[train_size:])
    return DataLoader(train_dataset, batch_size, shuffle=True), DataLoader(valid_dataset, batch_size)

sample_size = 1000
tau = 4
epochs = 5
lr = 0.01

model = nn.Sequential(
    nn.Linear(4, 10),
    nn.ReLU(),

    nn.Linear(10, 1)
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

time_steps, values = generate_data_from_sin(0, 0.2, size=sample_size, tau=tau)
features, labels = generate_features_and_labels(values, sample_size, tau)
train_loader, valid_loader = get_dataloader(features, labels, train_size=int(0.8 * sample_size), batch_size=16)

# torch.Size([1000, 4])
# torch.Size([1000])
print(features.shape)
print(labels.shape)


for epoch in range(epochs):
    model.train()
    train_loss_accu = 0
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_features)
        loss = loss_fn(predictions.squeeze(), batch_labels)
        loss.backward()
        optimizer.step()
        train_loss_accu += loss.item()

    model.eval()
    valid_loss_accu = 0
    with torch.no_grad():
        for batch_features, batch_labels in valid_loader:
            predictions = model(batch_features)
            loss = loss_fn(predictions.squeeze(), batch_labels)
            valid_loss_accu += loss.item()
    train_loss = train_loss_accu / len(train_loader)
    valid_loss = valid_loss_accu / len(valid_loader)
    print(f'第{epoch + 1} /{epochs}轮，训练损失：{train_loss:.4f}，测试损失：{valid_loss:.4f}')


def multi_step_predict(features, step_begin, step_num):
    data = features[step_begin]
    result = []
    for i in range(step_num):
        predicted = model(data).detach()
        result.append(predicted.squeeze())
        data = torch.cat((data[1:], predicted), dim=0)
    return torch.stack(result)

# 单步预测
step001_predi = model(features).detach().squeeze()

# 多步预测
step500_predi = multi_step_predict(features, step_begin=400, step_num=500)

# 可视化
plt.figure(figsize=(10, 5))
plt.plot(time_steps, values, label='Sine Wave with Noise', lw=0.8, ls='-', color='#2E7CEE')
plt.plot(time_steps[:sample_size], step001_predi, label='Predict 001 step(s)', lw=0.8, ls='-.', color='#FCC526')
plt.plot(time_steps[400:400 + 500], step500_predi, label='Predict 500 step(s)', lw=0.8, ls='-.', color='#E53E31')
plt.xlabel('time_steps')
plt.ylabel('values')
plt.legend()
plt.grid()
plt.show()
