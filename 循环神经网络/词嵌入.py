import torch
import torch.nn as nn

vocab_size = 10
embed_dim = 3

input_indices = torch.LongTensor([[1, 3, 4, 5],
                                  [4, 3, 2, 9]]) # [2, 4]

embedding_layer = nn.Embedding(vocab_size, embed_dim)
embedding_vectors = embedding_layer(input_indices) # 在输入张量的最后一个维度添加一个：embed_dim 新维度，使得每个词都会被表示为一个向量

print(embedding_vectors.shape)
print(embedding_vectors)