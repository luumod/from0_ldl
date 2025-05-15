import torch
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