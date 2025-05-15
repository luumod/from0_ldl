import torch
import torch.nn as nn
from time import time
import math

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