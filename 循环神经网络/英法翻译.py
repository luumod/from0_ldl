import os
import torch
import math
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from encoder_decoder import AbstractEncoder, AbstractDecoder, EncoderDecoder
from text_preprocessing import ST
from time import time
from collections import Counter

class Seq2SeqEncoder(AbstractEncoder):
    def __init__(self, vocab_size, embed_dim, hidden_num, num_layers, dropout=0.0):
        super(Seq2SeqEncoder, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size=hidden_num, num_layers=num_layers, dropout=dropout)

    def forward(self, input_seq, valid_lengths=None, **kwargs):
        '''
        将输入序列编码为中间表示
        :param input_seq: 输入张量，由词元索引组成，形状为（batch_size, seq_length）
        :param valid_lengths: 各序列的有效长度，形状为（batch_size, ） None表示所有序列的有效长度相同
        :return: 编码器的输出和最终的隐状态元组
        形状为 (output: (seq_length, batch_size, hidden_num)， state: (num_layers, batch_size, hidden_num))
        '''
        if input_seq.dim() != 2:
            raise ValueError(f'input_seq 应为二维张量')
        if input_seq.dtype != torch.long:
            input_seq = input_seq.long() # 转为LongTensor或者IntTensor用于 nn.Embedding

        # (batch_size, seq_length) --> (batch_size, seq_length, embed_dim) -> (seq_length, batch_size, embed_dim)
        embedded = self.embedding_layer(input_seq).permute(1, 0, 2,).contiguous()

        if valid_lengths is None:
            output, state = self.rnn(embedded) # 未显式提供隐状态，将自动创建全零张量
        else:
            # 压缩为无填充的紧密格式
            packed = pack_padded_sequence(
                input=embedded,
                lengths=valid_lengths.cpu(),
                enforce_sorted=False
            )
            output, state = self.rnn(packed)

            # 序列解包，转换为填充格式
            output, _ = pad_packed_sequence(output)

        return output, state #  其实重要的是state，因为解码层需要来自编码层输出的隐状态

class Seq2SeqDecoder(AbstractDecoder):
    def __init__(self, vocab_size, embed_dim, hidden_num, num_layers, dropout=0.0):
        super(Seq2SeqDecoder, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim  + hidden_num, hidden_size=hidden_num, num_layers=num_layers, dropout=dropout)
        self.output_layer = nn.Linear(hidden_num, vocab_size)
        self.hidden_num = hidden_num

    def init_state(self, enc_outputs, **kwargs):
        '''
        从编码器的输出中返回上下文向量，作为解码器的初始隐状态
        :param enc_outputs: 编码器的输出
        :return: 解码器的初始隐状态，形状为：(num_layers, batch_size, hidden_num)
        '''
        return enc_outputs[1]

    def forward(self, input_seq, state):
        '''
        执行序列的解码
        :param input_seq: 对于【解码层】的输入序列张量，由词元索引组成，形状为: (batch_size, seq_length)
        :param state: 对于【解码层】的初始隐状态（即解码器的输出隐状态），形状为：(num_layers, batch_size, hidden_num)
        :return: 解码器输出和更新后的隐状态元组
        '''
        if input_seq.dim() != 2:
            raise ValueError(f'input_seq 应为二维张量')
        if input_seq.dtype != torch.long:
            input_seq = input_seq.long()

        # (seq_length, batch_size, embed_dim)
        embedded = self.embedding_layer(input_seq).permute(1, 0, 2).contiguous()

        # (seq_length, batch_size, hidden_num)
        context = state[-1:].expand(embedded.shape[0], -1, -1)

        # (seq_length, batch_size, embed_dim + hidden_num)
        rnn_input = torch.cat([embedded, context], dim=2)

        output, state = self.rnn(rnn_input, state)
        output = self.output_layer(output)

        return (output, ), state # 重要的是 output，因为解码层只需要获取最后的输出即可，无需state

class SequenceLengthCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, label_smoothing=0.0):
        super(SequenceLengthCrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight,
                                                 size_average=size_average,
                                                 ignore_index=-100,
                                                 reduce=reduce,
                                                 reduction='none',
                                                 label_smoothing=label_smoothing)
    def forward(self, inputs, targets, valid_lengths):
        '''
        基于序列有效长度计算交叉熵损失
        :param inputs: 模型预测的输出，形状为：(seq_length, batch_size, vocab_size)
        :param targets: 目标标签，形状为： (batch_size, seq_length)
        :param valid_length: 各序列的有效长度，形状为：(batch_size, )
        :return: 掩码后损失的平均值
        '''
        inputs = inputs.permute(1, 2, 0) # (batch_size, vocab_size, seq_length)

        seq_length = targets.shape[1]

        # (batch_size, seq_length)
        # ([[True, True, True, True],
        # [True, True, False, False],
        # [False, False, False, False]])
        mask = torch.arange(seq_length, device=targets.device).unsqueeze(dim=0) < valid_lengths.unsqueeze(dim=1)

        losses = self.cross_entropy(inputs, targets) # (batch_size, seq_length)
        masked_mean_losses = (losses * mask.float()).mean(dim=1)

        return masked_mean_losses

def train_one_epoch(model, data_iter, optimizer, criterion, tgt_vocab, device):
    sos_index = tgt_vocab.get_index(ST.SOS)
    total_loss = 0.0
    total_tokens = 0

    model.train()
    start_time = time()
    for src, src_valid_len, tgt, tgt_valid_len in data_iter:
        optimizer.zero_grad()

        # src作为编码层输入
        src, tgt, tgt_valid_len = src.to(device), tgt.to(device), tgt_valid_len.to(device)

        # dec_input作为解码层的输入，形状 (batch_size, seq_length)
        dec_input = torch.cat([
            torch.full((tgt.shape[0], 1), sos_index, device=device), # (batch_size, 1) 每个都以SOS的索引开始，填充1个字符
            tgt[..., :-1] # 填充19个字符（去掉最后一个字符）
        ], dim=1)

        # ((seq_length, batch_size, vocab_size), )
        tgt_pred = model(src, dec_input, valid_lengths=src_valid_len)
        loss = criterion(inputs=tgt_pred[0], targets=tgt, valid_lengths=tgt_valid_len) # (batch_size, )

        loss.sum().backward()
        # 梯度裁剪
        clip_grad_norm_(model.parameters(), max_norm=2)
        optimizer.step()

        total_tokens += tgt_valid_len.sum().item()
        total_loss += loss.sum().item()

    avg_loss = total_loss / total_tokens
    tokens_per_sec = total_tokens / (time() - start_time)

    return avg_loss, tokens_per_sec

def forecast_greedy_search(model, src_sentence, src_vocab, tgt_vocab, device, max_length=20, record_attn_weights=False):
    pad_src_index = src_vocab.get_index(ST.PAD)
    sos_tgt_index = tgt_vocab.get_index(ST.SOS)
    eos_tgt_index = tgt_vocab.get_index(ST.EOS)

    # 对于输入预处理：词元化，编码之后进行填充
    src_tokens = src_vocab.encode([*src_sentence.lower().split(), ST.EOS]) #: [xx, xx, '<EOS>'] --> [48, 23, 3]
    src_tokens_pad_trunc = [
        *src_tokens[:max_length],
        *[pad_src_index] * (max_length - len(src_tokens))
    ] # (seq_length, ):  [48, 23, 3, 1, 1, 1, 1, 1, ...]

    # 组织逐时间步的输出
    output_tokens = []
    attn_weights = []

    model.eval()
    with torch.no_grad():
        # src_input: (batch_size=1, seq_length)
        src_input = torch.tensor([src_tokens_pad_trunc], dtype=torch.long, device=device)
        # 初始dec_input: (batch_size=1, 1)
        dec_input = torch.tensor(data=[[sos_tgt_index]], dtype=torch.long, device=device)
        # src_valid_length: (batch_size, )
        src_valid_length = torch.tensor([len(src_tokens)], device=device)

        for _ in range(max_length):
            output = model(src_input, dec_input, valid_lengths=src_valid_length)
            next_token = output[0][-1].argmax(dim=-1).item() # (seq_length, batch_size, 【vocab_size】)

            if next_token == eos_tgt_index:
                break
            if record_attn_weights:
                attn_weights.append(output[1].squeeze(0))

            output_tokens.append(next_token)

            # 叠加上一次的输出，预测新的结果
            dec_input = torch.cat(tensors=[dec_input, torch.tensor([[next_token]], device=device)],
                                  dim=1)  # (BATCH_SIZE=1, 1) -> (BATCH_SIZE=1, 2) -> ...

        tgt_sentence = ' '.join(tgt_vocab.decode(output_tokens))
        stack_attn_weights = torch.stack(attn_weights) if record_attn_weights else None
        return tgt_sentence, stack_attn_weights


def evaluate_bleu(candi_str, refer_strs, max_n_gram: int = 4, weights= None):
    """
    计算候选句子与参考句子之间的 BLEU 分数（基于空格分词）
    :param candi_str: 候选翻译/生成的文本
    :param refer_strs: 一个或多个参考翻译/标准文本
    :param max_n_gram: 最大 n-gram
    :param weights: n-gram 权重列表，默认为均匀权重 [0.25, 0.25, 0.25, 0.25]
    :return: BLEU 得分，范围从 0~1
    """
    weights = [1.0 / max_n_gram] * max_n_gram if weights is None else weights  # 设置默认权重

    if len(weights) != max_n_gram:  # 确保权重列表与最大 n-gram 匹配
        raise ValueError(f'n-gram 权重列表 weights 的长度必须等于 {max_n_gram=}，当前为{len(weights)}')

    candi_tokens: list[str] = candi_str.strip().split()  # 候选文本分词
    candi_len = len(candi_tokens)  # 候选文本长度
    refer_tokens_list: list[list[str]] = [ref.strip().split() for ref in refer_strs]  # 参考文本列表分词
    refer_len = min([len(ref_tokens) for ref_tokens in refer_tokens_list], key=lambda x: abs(x - candi_len))  # 参考文本长度

    # 计算短文本惩罚因子 (Brevity Penalty)
    if candi_len == 0:
        return 0.0  # 避免除以零错误
    bp = 1.0 if candi_len > refer_len else math.exp(1 - refer_len / candi_len)

    precisions = []  # 各阶 n-gram 精度列表
    for n in range(1, max_n_gram + 1):
        candi_ngrams = [' '.join(candi_tokens[i:i + n]) for i in range(len(candi_tokens) - n + 1)]  # 从候选文本中提取 n-gram
        if len(candi_ngrams) == 0:  # 如果没有 n-gram，则精度为小值 1e-10，避免 log(0) 问题
            precisions.append(1e-10)
            continue

        candi_counter = Counter(candi_ngrams)  # 统计候选文本中每个 n-gram 的出现次数
        max_matches = Counter()  # 计算匹配的 n-gram 数量 (对每个参考文本统计，然后取最大值)
        for ref_tokens in refer_tokens_list:
            ref_ngrams = [' '.join(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)]  # 从参考文本中提取 n-gram
            ref_counter = Counter(ref_ngrams)

            # 对每个 n-gram，取候选文本和参考文本中出现次数的最小值
            # 然后与当前最大匹配次数比较
            for ngram, count in candi_counter.items():
                max_matches[ngram] = max(max_matches[ngram], min(count, ref_counter[ngram]))

        matches = sum(max_matches.values())  # 计算匹配的 n-gram 总数
        precision = matches / sum(candi_counter.values()) if sum(candi_counter.values()) > 0 else 1e-10  # 计算精度
        precision = max(precision, 1e-10)

        precisions.append(precision)

    log_avg = sum(w * math.log(p) for w, p in zip(weights, precisions))  # 加权几何平均
    bleu = bp * math.exp(log_avg)

    return bleu

class TestSentence:
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

if __name__ == '__main__':
    from translation_dataset_loader import nmt_eng_fra_dataloader

    BATCH_SIZE = 128
    SEQ_LENGTH = 20
    EMBED_DIM = 256
    HIDDEN_NUM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.2
    LEARNING_RATE = 0.0005
    EPOCHS_NUM = 50
    TEST_INTERVAL = 1
    TEST_SENTENCES = TestSentence(src=["I like apples .",
                                       "She reads books regularly .",
                                       "They play soccer together .",
                                       "We studied French yesterday .",
                                       "The weather is beautiful today ."],
                                  tgt=[["J'aime les pommes .", "J'adore les pommes .", "Les pommes me plaisent .",
                                        "Je raffole des pommes .", "J'apprécie les pommes ."],
                                       ["Elle lit des livres régulièrement .", "Elle lit des livres souvent .",
                                        "Elle lit des livres fréquemment .", "Elle lit régulièrement des ouvrages ."],
                                       ["Ils jouent au football ensemble .", "Ils jouent au foot ensemble .",
                                        "Ils pratiquent le football ensemble .", "Ensemble, ils jouent au football ."],
                                       ["Nous avons étudié le français hier .", "Hier, nous avons étudié le français .",
                                        "Nous avons appris le français hier .", "Nous avons fait du français hier ."],
                                       ["Le temps est magnifique aujourd'hui .", "Il fait beau aujourd'hui .",
                                        "Le temps est splendide aujourd'hui .", "La météo est belle aujourd'hui ."]])

    data_iter, eng_vocab, fra_vocab = nmt_eng_fra_dataloader(BATCH_SIZE, SEQ_LENGTH, num_workers=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nmt_model = EncoderDecoder(encoder=Seq2SeqEncoder(len(eng_vocab), EMBED_DIM, HIDDEN_NUM, NUM_LAYERS, DROPOUT),
                               decoder=Seq2SeqDecoder(len(fra_vocab), EMBED_DIM, HIDDEN_NUM, NUM_LAYERS, DROPOUT),
                               device=device)  # 使用默认的模型参数初始化方法，不手动初始化
    optimizer = torch.optim.Adam(nmt_model.parameters(), lr=LEARNING_RATE)
    criterion = SequenceLengthCrossEntropyLoss()

    # 模型检查点路径
    checkpoint_path = './models/translation_model.pth'
    start_epoch = 0
    # 加载检查点（如果存在）
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        nmt_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
        print(f"加载检查点，从第 {start_epoch} 个周期继续训练")
    else:
        os.mkdir('./models')

    src_sentence = "you are the chosen one. "
    forecast_fra, _ = forecast_greedy_search(nmt_model, src_sentence, eng_vocab, fra_vocab, device)
    print(f'{src_sentence} --> {forecast_fra}')

    # best_loss = float('inf')
    # for epoch in range(start_epoch, EPOCHS_NUM):
    #     loss, speed = train_one_epoch(nmt_model, data_iter, optimizer, criterion, fra_vocab, device)
    #     print(f'第 {epoch + 1:03} 轮：损失为 {loss:.3f}，速度为 {speed:.1f} tokens/sec')
    #
    #     if loss < best_loss:
    #         best_loss = loss
    #         # 保存最优模型
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': nmt_model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': loss,
    #         }, checkpoint_path)
    #
    #     if (epoch + 1) % TEST_INTERVAL == 0:
    #         for eng, fra in zip(TEST_SENTENCES.src, TEST_SENTENCES.tgt):
    #             forecast_fra, _ = forecast_greedy_search(nmt_model, eng, eng_vocab, fra_vocab, device)
    #             print(f'INFO: '
    #                   f'{eng.ljust(max(map(len, TEST_SENTENCES.src)))} '
    #                   f'→ (BLEU={evaluate_bleu(forecast_fra, fra, max_n_gram=3):.2f}) {forecast_fra}')