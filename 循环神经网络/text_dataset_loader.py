import torch
from text_preprocessing import get_vocab_corpus_from_timemachine
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, corpus, seq_length):
        self.corpus = corpus
        self.seq_length = seq_length

    def __len__(self):
        return len(self.corpus) - self.seq_length

    def __getitem__(self, idx):
        return (torch.tensor(self.corpus[idx:idx+self.seq_length]),
                torch.tensor(self.corpus[idx+1:idx+self.seq_length+1]))

def timemachine_data_loader(batch_size, seq_length, shuffle=False, max_token_num=10000):
    vocab, corpus = get_vocab_corpus_from_timemachine(token_type='char', max_token_num=max_token_num)
    data_iter = DataLoader(TextDataset(corpus, seq_length), batch_size=batch_size, shuffle=shuffle, drop_last=True) # 丢弃不完整的最后一个批次
    return data_iter, vocab


