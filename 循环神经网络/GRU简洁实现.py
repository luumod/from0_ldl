from text_dataset_loader import timemachine_data_loader
import torch
import math
from time import time
import torch.nn as nn
import torch.nn.functional as F

class RnnFrameWork(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        '''
        :param rnn_layer: 内置rnn层；nn.RNN, nn.GRU, nn.LSTM
        :param vocab_size: 词表大小（用于独热编码）
        '''
        super(RnnFrameWork, self).__init__()
        self.rnn_layer = rnn_layer
        self.rnn_direction_num = 2 if self.rnn_layer.bidirectional else 1
        self.rnn_hidden_num = self.rnn_layer.hidden_size # 512
        self.fc = nn.Linear(self.rnn_hidden_num * self.rnn_direction_num, vocab_size) # [512, 28]
        self.vocab_size = vocab_size

    def forward(self, inputs, states):
        '''
        :param inputs: 原始输入
        :param states: 上一步的隐状态
        :return: outputs输出 与 new_states隐状态的更新
        '''
        # inputs: [35, 32, 28]
        X = F.one_hot(inputs.T, self.vocab_size).type(torch.float32)
        states = states if isinstance(self.rnn_layer, nn.LSTM) else states[0]

        # 调用rnn_layer实例并且前向传播计算
        # outputs: [35, 32, 512], new_states: [1, 32, 512]
        # H_t = X_t * W_ih + H_t-1 * W_hh + b_h
        outputs, new_states = self.rnn_layer(X, states)

        # new_states: ([1, 32, 512], )
        #
        new_states = new_states if isinstance(self.rnn_layer, nn.LSTM) else (new_states, )

        # fc( [35 * 32, 512] ) ---> [1120, 512] * [512, 28] --> [1120, 28]
        logits = self.fc(outputs.reshape(-1, outputs.shape[-1]))

        return logits, new_states

    def init_hidden_states(self, batch_size, device):
        shape = (self.rnn_layer.num_layers * self.rnn_direction_num, batch_size, self.rnn_hidden_num)
        hidden_state = torch.zeros(shape, device=device)
        if isinstance(self.rnn_layer, nn.LSTM):
            cell_state = hidden_state.clone()
            return hidden_state, cell_state
        else:
            return (hidden_state, )

def forecast_chars(prefix, num, model, vocab, device):
    states = model.init_hidden_states(batch_size=1, device=device)
    outputs = [vocab.get_index(prefix[0])]

    # 预热期：首先生成prefix对应的索引编码列表：[3,5,7,2,2,1,...]
    for y in prefix[1:]:
        _, states = model(torch.tensor(outputs[-1:], device=device).unsqueeze(0), states)
        outputs.append(vocab.get_index(y))

    # 预测期：然后接着生成num个预测的索引编码
    for _ in range(num):
        y, states = model(torch.tensor(outputs[-1:], device=device).unsqueeze(0), states)
        outputs.append(torch.argmax(y, dim=1).item())

    # 统一解码后输出预测文本
    return ''.join(vocab.decode(outputs))

def clip_gradients(model, max_norm):
    params = [p for p in model.parameters() if p.requires_grad] if isinstance(model, nn.Module) else model.params

    grad_12_norm = torch.norm(torch.cat([p.grad.flatten() for p in params]), 2)
    if grad_12_norm > max_norm:
        for p in params:
            p.grad.mul_(max_norm / grad_12_norm)


def train_one_epoch(model, data_iter, loss_fn, optimizer, device, shuffle):
    states = None
    total_tokens = 0
    total_loss = 0.0
    start_time = time()

    for features, labels in data_iter:
        '''
        features: [batch_size, seq_length] --> [32, 35]
        labels:   [batch_size, seq_length] --> [32, 35]
        '''
        features = features.to(device)
        # 将labels展平: [32*35, ]，使其能够适应model(x)预测后的输出，进行损失的计算:
        labels = labels.T.flatten().to(device)

        if shuffle or states is None:
            # 首先初始化一个空的状态
            states = model.init_hidden_states(features.shape[0], device)
        else:
            states = tuple(s.detach() for s in states)

        '''
        outputs: [batch_size * seq_length, d] --> [1120, 28]
        states: 更新后的隐藏状态
        '''
        outputs, states = model(features, states)

        # loss: [1120, 28]
        loss = loss_fn(outputs, labels)

        if isinstance(optimizer, torch.optim.Optimizer):
            optimizer.zero_grad()
            loss.backward()
            clip_gradients(model, max_norm=1)
            optimizer.step()
        else:
            loss.backward()
            clip_gradients(model, max_norm=1)
            optimizer(features.shape[0])

        batch_tokens = labels.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    perplexity = math.exp(total_loss / total_tokens)
    tokens_per_sec = total_tokens / (time() - start_time)

    return perplexity, tokens_per_sec



if __name__ == '__main__':
    batch_size = 32
    seq_length = 35
    hidden_num = 512
    num_epochs = 50
    lr = 0.7
    is_shuffle = False
    forcast_interval = 10
    prefix_string = 'time traveller'

    data_iter, vocab = timemachine_data_loader(batch_size, seq_length)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gru_core = nn.GRU(input_size=len(vocab), hidden_size=hidden_num)
    rnn = RnnFrameWork(rnn_layer=gru_core, vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.SGD(rnn.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        ppl, speed = train_one_epoch(rnn, data_iter, loss_fn, optimizer, device, is_shuffle)
        print(f'第{epoch + 1}轮：困惑度为 {ppl}，速度为 {speed}')

        if (epoch + 1) % forcast_interval == 0:
            with torch.no_grad():
                prediction = forecast_chars(prefix_string, 50, rnn, vocab, device)
                print(f'预测结果：{prediction}')





