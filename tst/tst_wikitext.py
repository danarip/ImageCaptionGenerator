import torchtext
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

train_iter = WikiText2(split='train')

text = "dotan dana \n pico"
lines = text.split(sep="\n")
print(lines)
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, iter(lines)), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

print(vocab)

itos = vocab.vocab.get_itos()
print(itos[0])
print(itos[1])
print(itos[2])
#for i, token in enumerate(lines[0]):
#    print(f'{itos[token]}')