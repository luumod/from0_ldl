import torch
from time import time
import math
import torch.nn.functional as F
import torch.nn as nn
from text_dataset_loader import timemachine_data_loader
from typing import Tuple

class LSTMScratch:
    def __init__(self, vocab_size, hidden_num, device):
        self.__vocab_size = vocab_size
        self.__hidden_num = hidden_num

        self.params = self.__init_params(device)

    def __init_params(self, device):
        # 遗忘门参数
        W_xf = torch.randn((self.__vocab_size, self.__hidden_num), device=device) * 0.01
        W_hf = torch.randn((self.__hidden_num, self.__hidden_num), device=device) * 0.01
        b_f = torch.zeros(self.__hidden_num, device=device)

        # 输入门参数
        W_xi = torch.randn((self.__vocab_size, self.__hidden_num), device=device) * 0.01
        W_hi = torch.randn((self.__hidden_num, self.__hidden_num), device=device) * 0.01
        b_i = torch.zeros(self.__hidden_num, device=device)

        # 输出门参数
        W_xo = torch.randn((self.__vocab_size, self.__hidden_num), device=device) * 0.01
        W_ho = torch.randn((self.__hidden_num, self.__hidden_num), device=device) * 0.01
        b_o = torch.zeros(self.__hidden_num, device=device)

        # 候选记忆元参数
        W_xc = torch.randn((self.__vocab_size, self.__hidden_num), device=device) * 0.01
        W_hc = torch.randn((self.__hidden_num, self.__hidden_num), device=device) * 0.01
        b_c = torch.zeros(self.__hidden_num, device=device)

        # 输出层参数
        W_hq = torch.randn((self.__hidden_num, self.__vocab_size), device=device) * 0.01
        b_q = torch.zeros(self.__vocab_size, device=device)

        return (W_xf.requires_grad_(),
                W_hf.requires_grad_(),
                b_f.requires_grad_(),
                W_xi.requires_grad_(),
                W_hi.requires_grad_(),
                b_i.requires_grad_(),
                W_xo.requires_grad_(),
                W_ho.requires_grad_(),
                b_o.requires_grad_(),
                W_xc.requires_grad_(),
                W_hc.requires_grad_(),
                b_c.requires_grad_(),
                W_hq.requires_grad_(),
                b_q.requires_grad_())

    def init_hidden_states(self, batch_size: int, device: torch.device | str):
        """初始化隐状态，并用元组组织"""
        hidden_state = torch.zeros((batch_size, self.__hidden_num), device=device)
        memory_cell = torch.zeros((batch_size, self.__hidden_num), device=device)

        return hidden_state, memory_cell

    @staticmethod
    def __lstm_step(inputs, states, params):
        """
        RNN 的一个时间步内的隐状态计算
        :param inputs: 形状为：(SEQ_LENGTH, BATCH_SIZE, VOCAB_SIZE)
        :param states: 隐状态元组，其中的隐状态形状为：(BATCH_SIZE, HIDDEN_NUM)
        :param params: 模型参数元组
        :return: 由计算结果与隐状态元组组成的元组
        """
        W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
        H_t, memory_cell = states
        outputs_temp = []

        for X_t in inputs:
            # 遗忘门
            gate_forget = torch.sigmoid((X_t @ W_xf) + (H_t @ W_hf) + b_f)
            # 输入门
            gate_input = torch.sigmoid((X_t @ W_xi) + (H_t @ W_hi) + b_i)
            # 输出门
            gate_output = torch.sigmoid((X_t @ W_xo) + (H_t @ W_ho) + b_o)
            # 候选记忆元
            memory_cell_candidate = torch.tanh((X_t @ W_xc) + (H_t @ W_hc) + b_c)
            # 最终记忆元
            memory_cell = gate_forget * memory_cell + gate_input * memory_cell_candidate
            # 隐状态
            H_t = gate_output * torch.tanh(memory_cell)  # state: (BATCH_SIZE, HIDDEN_NUM)
            # 输出：[batch_size, vocab_size] --> [32, 28]
            output_layer = H_t @ W_hq + b_q
            outputs_temp.append(output_layer)

        outputs = torch.cat(outputs_temp, dim=0)
        # 每一次时间步的输出都会包含两个更新：隐状态、记忆元
        out_states = (H_t, memory_cell)

        return outputs, out_states

    def __call__(self, inputs, states):
        inputs = F.one_hot(inputs.T, self.__vocab_size).type(torch.float32)
        return self.__lstm_step(inputs, states=states, params=self.params)


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
    lstm = LSTMScratch(vocab_size=len(vocab ), hidden_num=HIDDEN_NUM, device=device)
    optimizer = torch.optim.SGD(lstm.params,lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        ppl, speed = train_one_epoch(lstm, data_iter, loss_fn, optimizer, device, is_shuffle)
        print(f'第{epoch + 1}轮：困惑度为 {ppl}，速度为 {speed} (tokens/sec)')

        if (epoch + 1) % forcast_interval == 0:
            with torch.no_grad():
                prediction = forecast_chars(prefix_string, 50, lstm, vocab, device)
                print(f'预测结果：{prediction}')

