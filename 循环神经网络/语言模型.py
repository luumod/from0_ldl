from text_preprocessing import get_vocab_corpus_from_timemachine, tokenize, get_timemachine_lines, generate_bigram_frequencies, generate_trigram_frequencies
import matplotlib.pyplot as plt

vocab, _ = get_vocab_corpus_from_timemachine(token_type='word')

token_freqs = vocab.valid_token_freqs
sorted_token_freqs = sorted(token_freqs.items(), key=lambda pair: pair[1], reverse=True)

print(*sorted_token_freqs[:10], sep='\n')

flat_tokens = [token for lines in [tokenize(line) for line in get_timemachine_lines()] for token in lines]

bigram = generate_bigram_frequencies(flat_tokens)
trigram = generate_trigram_frequencies(flat_tokens)

plt.plot([pair[1] for pair in sorted_token_freqs], label='1-gram')
plt.plot([pair[1] for pair in bigram], label='2-gram')
plt.plot([pair[1] for pair in trigram], label='3-gram')

plt.xlabel('n-gram frequency rank')
plt.ylabel('frequency')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()