import os
import torch
from text_preprocessing import tokenize, Vocabulary, ST

def process_eng_fra_dataset(normal_unicode = None, lowercase=True, max_sample_pair_num = None):
    dataset = 'eng_fra.txt'
    encoding = 'UTF-8'
    source_tab_target = []

    if not os.path.exists(dataset):
        import io, re, requests, zipfile, unicodedata
        temp_dataset = 'fra.txt'
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

    with open(dataset, 'r', encoding=encoding) as dataset_file:
        if max_sample_pair_num:
            # 获得规定的最大样本数
            source_tab_target = dataset_file.readlines()[:max_sample_pair_num]
        else:
            source_tab_target = dataset_file.readlines()

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
    :param max_simple_pair_num: 最大样本数
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
