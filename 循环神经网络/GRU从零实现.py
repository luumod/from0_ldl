import torch
from time import time
import math
import torch.nn.functional as F
import torch.nn as nn
from text_dataset_loader import timemachine_data_loader
from typing import Tuple

class GruScratch:
    def __init__(self, vocab_size, hidden_num, device):
        self.__vocab_size = vocab_size
        self.__hidden_num = hidden_num

        self.params = self.__init_params(device)

    def __init_params(self, device):
        # 重置门参数
        W_xr = torch.randn((self.__vocab_size, self.__hidden_num), device=device) * 0.01
        W_hr = torch.randn((self.__hidden_num, self.__hidden_num), device=device) * 0.01
        b_r = torch.zeros(self.__hidden_num, device=device)

        # 更新门参数
        W_xz = torch.randn((self.__vocab_size, self.__hidden_num), device=device) * 0.01
        W_hz = torch.randn((self.__hidden_num, self.__hidden_num), device=device) * 0.01
        b_z = torch.zeros(self.__hidden_num, device=device)

        # 候选隐状态参数
        W_xh = torch.randn((self.__vocab_size, self.__hidden_num), device=device) * 0.01
        W_hh = torch.randn((self.__hidden_num, self.__hidden_num), device=device) * 0.01
        b_h = torch.zeros(self.__hidden_num, device=device)

        # 输出层参数
        W_hq = torch.randn((self.__hidden_num, self.__vocab_size), device=device) * 0.01
        b_q = torch.zeros(self.__vocab_size, device=device)

        return (W_xr.requires_grad_(),
                W_hr.requires_grad_(),
                b_r.requires_grad_(),
                W_xz.requires_grad_(),
                W_hz.requires_grad_(),
                b_z.requires_grad_(),
                W_xh.requires_grad_(),
                W_hh.requires_grad_(),
                b_h.requires_grad_(),
                W_hq.requires_grad_(),
                b_q.requires_grad_())

    def init_hidden_states(self, batch_size: int, device: torch.device | str):
        """初始化隐状态，并用元组组织"""
        return (torch.zeros((batch_size, self.__hidden_num), device=device),)

    @staticmethod
    def __rnn_step(inputs, states, params):
        """
        RNN 的一个时间步内的隐状态计算
        :param inputs: 形状为：(SEQ_LENGTH, BATCH_SIZE, VOCAB_SIZE)
        :param states: 隐状态元组，其中的隐状态形状为：(BATCH_SIZE, HIDDEN_NUM)
        :param params: 模型参数元组
        :return: 由计算结果与隐状态元组组成的元组
        """
        W_xr, W_hr, b_r, W_xz, W_hz, b_z, W_xh, W_hh, b_h, W_hq, b_q = params
        H_t, = states
        outputs_temp = []

        for X_t in inputs:
            # 重置门
            gate_reset = torch.sigmoid((X_t @ W_xr) + (H_t @ W_hr) + b_r)
            # 更新门
            gate_update = torch.sigmoid((X_t @ W_xz) + (H_t @ W_hz) + b_z)
            # 候选隐状态
            hidden_candidate = torch.tanh((X_t @ W_xh) + ((gate_reset * H_t) @ W_hh) + b_h)
            # 最终隐状态
            H_t = (gate_update * H_t) + (1 - gate_update) * hidden_candidate
            output_layer = H_t @ W_hq + b_q
            outputs_temp.append(output_layer)

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
    num_epochs = 50
    lr = 0.7
    is_shuffle = False
    forcast_interval = 10
    prefix_string = 'time traveller'

    data_iter, vocab = timemachine_data_loader(BATCH_SIZE, SEQ_LENGTH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gru = GruScratch(vocab_size=len(vocab ), hidden_num=HIDDEN_NUM, device=device)
    optimizer = torch.optim.SGD(gru.params,lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        ppl, speed = train_one_epoch(gru, data_iter, loss_fn, optimizer, device, is_shuffle)
        print(f'第{epoch + 1}轮：困惑度为 {ppl}，速度为 {speed} (tokens/sec)')

        if (epoch + 1) % forcast_interval == 0:
            with torch.no_grad():
                prediction = forecast_chars(prefix_string, 50, gru, vocab, device)
                print(f'预测结果：{prediction}')

