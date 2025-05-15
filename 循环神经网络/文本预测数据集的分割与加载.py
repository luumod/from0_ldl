import torch
from text_preprocessing import get_vocab_corpus_from_timemachine
from torch.utils.data import Dataset, DataLoader

class LanguageModelDataGenerator:
    def __init__(self, corpus, seq_length, batch_size, shuffle):
        self.corpus = corpus
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.samples_num = len(corpus) -seq_length

    def __len__(self):
        return self.samples_num // self.batch_size

    def __iter__(self):
        indices = torch.arange(self.samples_num)
        if self.shuffle:
            indices = indices[torch.randperm(self.samples_num)]

        for i in range(0, self.samples_num, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            features = [self.corpus[idx:idx + self.seq_length]
                        for idx in batch_indices]
            labels = [self.corpus[idx+1:idx+self.seq_length+1]
                      for idx in batch_indices]
            yield torch.tensor(features),torch.tensor(labels)

# def timemachine_data_loader(batch_size,seq_length, shuffle=False, max_token_num=10000):
#     vocab, corpus = get_vocab_corpus_from_timemachine(token_type='char', max_token_num=max_token_num)
#     data_iter = LanguageModelDataGenerator(corpus, seq_length, batch_size, shuffle)
#     return data_iter, vocab

class TextDataset(Dataset):
    def __init__(self, corpus, seq_length):
        self.corpus = corpus
        self.seq_length = seq_length

    def __len__(self):
        return len(self.corpus) - self.seq_length

    def __getitem__(self, idx):
        return (torch.tensor(self.corpus[idx:idx+self.seq_length]),
                torch.tensor(self.corpus[idx+1:idx+self.seq_length+1]))

def timemachine_data_loader(batch_size,seq_length, shuffle=False, max_token_num=10000):
    vocab, corpus = get_vocab_corpus_from_timemachine(token_type='char', max_token_num=max_token_num)
    data_iter = DataLoader(TextDataset(corpus, seq_length), batch_size=batch_size, shuffle=shuffle, drop_last=True) # 丢弃不完整的最后一个批次
    return data_iter, vocab

if __name__ == '__main__':
    data_iter, vocab = timemachine_data_loader(batch_size=5, seq_length=10)

    for f, l in data_iter:
        print(f'特征：{f.tolist()}')
        print(f'标签：{l.tolist()}')

        print(f'第 1 个特征解码：{"".join(vocab.decode(f[0].tolist()))!r}')
        break
    print(f'每个小批量中的样本数：{data_iter.batch_size}')
    print(f'批量总数：{len(data_iter)}')

