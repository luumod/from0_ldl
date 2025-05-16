from abc import ABC as _ABC, abstractmethod as _abstractmethod
from torch import nn as _nn

class AbstractEncoder(_ABC, _nn.Module):
    def __init__(self):
        super(AbstractEncoder, self).__init__()

    @_abstractmethod
    def forward(self, input_seq, **kwargs):
        raise NotImplementedError

class AbstractDecoder(_ABC, _nn.Module):
    def __init__(self):
        super(AbstractDecoder, self).__init__()

    @_abstractmethod
    def init_state(self, enc_outputs, **kwargs):
        raise NotImplementedError

    @_abstractmethod
    def forward(self, input_seq, state):
        raise NotImplementedError


class EncoderDecoder(_nn.Module):
    def __init__(self, encoder, decoder, device):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

    def forward(self, enc_input_seq, dec_input_seq, **kwargs):
        '''
        :param enc_input_seq: 经词嵌入后输入到编码层的序列（原始文本）：(batch_size, seq_length)
        :param dec_input_seq: 经词嵌入后输入到解码层的序列（翻译后文本）：(batch_size, seq_length(训练) | 1(预测)) 每一个样本都以SOS对应的索引开始，长度为seq_length
        :return: 经解码层输出后的序列，形状：(seq_length, batch_size, vocab_size)
        '''
        enc_outputs = self.encoder(enc_input_seq, **kwargs)
        dec_state = self.decoder.init_state(enc_outputs, **kwargs) # 获取来自编码层的state
        dec_outputs, _ = self.decoder(dec_input_seq, dec_state)
        return dec_outputs



