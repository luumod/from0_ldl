import torch
from time import time
import math
import torch.nn.functional as F
import torch.nn as nn
from text_dataset_loader import timemachine_data_loader
from typing import Tuple

# X = torch.normal(0, 1, size=(16, 10))
# W_xh = torch.normal(0, 1, size=(10, 4))
# H_ex = torch.normal(0, 1, size=(16, 8))
# W_hh = torch.normal(0, 1, size=(8, 4))
#
# o_1 = X @ W_xh + H_ex @ W_hh # 当前输入加权与历史状态加权的叠加
# o_2 = torch.cat((X, H_ex), dim=1) @ torch.cat((W_xh, W_hh), dim=0)  # 当前输入在序列样本量上的历史状态延长与权重延长的加权
#
# print(f'一次时间步计算后的输出 o_1 维度为：{(*o_1.shape,)}')
# print(f'一次时间步计算后的输出 o_2 维度为：{(*o_2.shape,)}')

batch_size = 32
seq_length = 35
data_iter, vocab = timemachine_data_loader(batch_size, seq_length)

# 独热编码
token = 'a'
index = vocab.get_index(token)
hot_item = F.one_hot(torch.tensor(index), len(vocab))
index_from_hot = torch.argmax(hot_item, dim=0).item()

print(f'词元{token}的索引值为：{index}，独热编码为：\n{hot_item}')
print(f'独热编码的结果重新转换为索引：{index_from_hot}')

# [32, 35]
batch = torch.zeros(batch_size, seq_length, dtype=torch.long)
batch_t = batch.T
hot_batch = F.one_hot(batch_t, len(vocab))
print(f'每批次经独热编码后的形状：{hot_batch.shape}')
from typing import Protocol


class RnnProtocol(Protocol):
    params: Tuple[torch.Tensor, ...]

    def init_hidden_states(self, batch_size, device):
        ...

class RnnScratch:
    def __init__(self, vocab_size, hidden_num, device):
        self.__vocab_size = vocab_size
        self.__hidden_num = hidden_num

        self.params = self.__init_params(device)

    def __init_params(self, device):
        # 初始化参数，并设置 requires_grad=True
        w_input2hidden = torch.randn((self.__vocab_size, self.__hidden_num), device=device) * 0.01
        w_hidden2hidden = torch.randn((self.__hidden_num, self.__hidden_num), device=device) * 0.01
        b_hidden = torch.zeros(self.__hidden_num, device=device)

        w_hidden2output = torch.randn((self.__hidden_num, self.__vocab_size), device=device) * 0.01
        b_output = torch.zeros(self.__vocab_size, device=device)

        return (w_input2hidden.requires_grad_(),
                w_hidden2hidden.requires_grad_(),
                b_hidden.requires_grad_(),
                w_hidden2output.requires_grad_(),
                b_output.requires_grad_())

    def init_hidden_states(self, batch_size: int, device: torch.device | str):
        """初始化隐状态，并用元组组织"""
        return torch.zeros((batch_size, self.__hidden_num), device=device),

    @staticmethod
    def __rnn_step(inputs, states, params):
        """
        RNN 的一个时间步内的隐状态计算
        :param inputs: 形状为：(SEQ_LENGTH, BATCH_SIZE, VOCAB_SIZE)
        :param states: 隐状态元组，其中的隐状态形状为：(BATCH_SIZE, HIDDEN_NUM)
        :param params: 模型参数元组
        :return: 由计算结果与隐状态元组组成的元组
        """
        w_input2hidden, w_hidden2hidden, b_hidden, w_hidden2output, b_output = params
        H_t, = states
        outputs_temp = []

        for X_t in inputs:
            H_t = torch.tanh(
                X_t @ w_input2hidden + H_t @ w_hidden2hidden + b_hidden
            )
            O_t = H_t @ w_hidden2output + b_output
            outputs_temp.append(O_t)

        outputs = torch.cat(outputs_temp, dim=0)
        out_states = (H_t,)

        return outputs, out_states

    def __call__(self, inputs, states):
        inputs = F.one_hot(inputs.T, self.__vocab_size).type(torch.float32)
        return self.__rnn_step(inputs, states=states, params=self.params)


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
    BATCH_SIZE = 32
    SEQ_LENGTH = 35
    HIDDEN_NUM = 512
    num_epochs = 500
    lr = 0.7
    is_shuffle = False
    forcast_interval = 10
    prefix_string = 'time traveller'

    data_iter, vocab = timemachine_data_loader(BATCH_SIZE, SEQ_LENGTH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rnn = RnnScratch(vocab_size=len(vocab ), hidden_num=HIDDEN_NUM, device=device)
    optimizer = torch.optim.SGD(rnn.params,lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        ppl, speed = train_one_epoch(rnn, data_iter, loss_fn, optimizer, device, is_shuffle)
        print(f'第{epoch + 1}轮：困惑度为 {ppl}，速度为 {speed} (tokens/sec)')

        if (epoch + 1) % forcast_interval == 0:
            with torch.no_grad():
                prediction = forecast_chars(prefix_string, 50, rnn, vocab, device)
                print(f'预测结果：{prediction}')

