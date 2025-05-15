import os

import torch

from text_preprocessing import tokenize, Vocabulary, ST
import matplotlib.pyplot as plt

def process_eng_fra_dataset(normal_unicode = None, lowercase=True, max_sample_pair_num = None):
    dataset = 'eng_fra.txt'
    encoding = 'UTF-8'
    source_tab_target = []

    if not os.path.exists(dataset):
        import io, re, requests, zipfile, unicodedata
        temp_dataset  = 'fra.txt'
        url = r'https://www.manythings.org/anki/fra-eng.zip'

        print(f'下载{url}...')
        response = requests.get(url, headers={'User-Agent': '...'}, timeout=30)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extract(temp_dataset)

        with open(temp_dataset, 'r', encoding=encoding) as temp_dataset_file:
            lines = temp_dataset_file.readlines()
            for line in lines:
                if lowercase:
                    line = line.lower()
                if normal_unicode:
                    line = unicodedata.normalize(normal_unicode, line)

                # 在单词和标点符号之间插入空格
                line = re.sub(r'([!.,;?"])', r' \1 ', line)
                line = line.replace('\u00A0', ' ')\
                            .replace('\u202F', ' ')\
                            .replace('’', "'")\
                            .replace(' \t', '\t')
                line = re.sub(r' +', r' ', line)

                eng, fra, *_ = line.strip().split(sep='\t')
                source_tab_target.append(f'{eng}\t{fra}')

        if os.path.exists(temp_dataset):
            os.remove(temp_dataset)

        with open(dataset, 'w', encoding=encoding) as dataset_file:
            dataset_file.write('\n'.join(source_tab_target))

    with open(dataset, 'r', encoding=encoding) as cachefile:
        if max_sample_pair_num:
            source_tab_target = cachefile.readlines()[:max_sample_pair_num]
        else:
            source_tab_target = cachefile.readlines()

    sources, targets = [], []
    for line in source_tab_target:
        source, target = line.strip().split(sep='\t')
        sources.append(source)
        targets.append(target)

    return sources, targets

def get_encoded_padded_tensor(lines, vocab, padding_length):
    '''
    将每个句子进行【编码】，【填充】或者【截断】，确保序列长度相同
    :param lines: 每个要处理的句子（英语 或者 法语）
    :param vocab: 对应语言的词汇表
    :param padding_length: 要填充到的目标长度
    :return:
    '''
    encoded_pad = vocab.get_index(ST.PAD)

    # 每个句子编码：eg. [[48, 4, 3], ... ]
    lines_encoded = [vocab.encode(token) + [vocab.get_index(ST.EOS)] for token in lines]

    # 将每个编码后的句子长度适应到到指定长度。
    # eg. lines_padded = [[48, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], ...]
    lines_padded = [encoded_line[:padding_length] + [encoded_pad] * (padding_length - len(encoded_line))
                    for encoded_line in lines_encoded]

    tensor_lines_padded = torch.tensor(lines_padded)

    # 计算每个句子的有效长度：eg. [3, 3, ....]
    length_without_padding = torch.sum(torch.ne(tensor_lines_padded, encoded_pad), dim=1)

    return tensor_lines_padded, length_without_padding

from torch.utils.data import TensorDataset, DataLoader

# 构建数据集
def nmt_eng_fra_dataloader(batch_size, seq_length, num_workers = 4,max_simple_pair_num = None):
    '''
    加载 英语 - 法语 翻译数据集
    :param batch_size: 每次读取的批量大小，例如batch_size=2，则意味着每次取出两个【英语-法语】句子对
    :param seq_length: 规定的每个句子的最大长度，例如seq_length=20，如果某句子编码后长度不足20，则填充 ST.PAD
    :param num_workers: 用于DataLoader
    :param max_simple_pair_num:
    :return:
    '''
    # 得到英语、法语的所有句子： ['go .', 'go .', 'go .',  ...]
    eng_list, fra_list = process_eng_fra_dataset(normal_unicode='NFC', lowercase=True,
                                                 max_sample_pair_num=max_simple_pair_num)

    # 按【词】进行词元化处理：[['go', '.'], ['go', '.'] ...]
    eng_tokenized: list[list[str]] = [tokenize(string, token_type='word') for string in eng_list]
    fra_tokenized: list[list[str]] = [tokenize(string, token_type='word') for string in fra_list]
    special_tokens_nmt = (ST.UNK, ST.PAD, ST.SOS, ST.EOS)

    # 得到双语词汇表
    eng_vocab = Vocabulary([word for string in eng_tokenized for word in string], special_tokens_nmt, min_freq=2)
    fra_vocab = Vocabulary([word for string in fra_tokenized for word in string], special_tokens_nmt, min_freq=2)

    # 获取编码后的所有样本， 以及对应的有效序列长度
    eng_tensor, eng_valid_len = get_encoded_padded_tensor(eng_tokenized, eng_vocab, seq_length)
    fra_tensor, fra_valid_len = get_encoded_padded_tensor(fra_tokenized, fra_vocab, seq_length)

    dataset = TensorDataset(eng_tensor, eng_valid_len, fra_tensor, fra_valid_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader, eng_vocab, fra_vocab

if __name__ == '__main__':
    data_iter, eng_vocab, fra_vocab = nmt_eng_fra_dataloader(batch_size=2, seq_length=20)

    print(f'英文词表大小：{len(eng_vocab)}')
    print(f'法文词表大小：{len(fra_vocab)}\n')

    for eng_encoded_lines, eng_valid_len, fra_encoded_lines, fra_valid_len in data_iter:
        print(f'英文句子编码值：{eng_encoded_lines.tolist()}')
        print(f'英文句子有效长度：{eng_valid_len.tolist()}')
        print(f'英文句子解码（去除填充词元）：'
              f'{[" ".join(eng_vocab.decode(line[:length].tolist())) for line, length in zip(eng_encoded_lines, eng_valid_len)]}\n')

        print(f'法文句子编码值：{fra_encoded_lines.tolist()}')
        print(f'法文句子有效长度：{fra_valid_len.tolist()}')
        print(f'法文句子解码（去除填充词元）：'
              f'{[" ".join(fra_vocab.decode(line[:length].tolist())) for line, length in zip(fra_encoded_lines, fra_valid_len)]}')

        break

    # 运行结果
    # 英文词表大小：11881
    # 法文词表大小：20839
    #
    # 英文句子编码值：[[8, 1039, 15, 8, 1038, 643, 7, 319, 567, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                 [20, 1236, 8, 6112, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    # 英文句子有效长度：[11, 6]
    # 英文句子解码（去除填充词元）：['the leaves of the trees began to turn red . <EOS>', "don't feed the pigeons . <EOS>"]
    #
    # 法文句子编码值：[[24, 1869, 36, 1487, 5077, 11, 646, 2378, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                 [10, 8366, 8, 24, 7836, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    # 法文句子有效长度：[10, 7]
    # 法文句子解码（去除填充词元）：['les feuilles des arbres commencèrent à devenir rouges . <EOS>',
    #                             'ne nourrissez pas les pigeons . <EOS>']