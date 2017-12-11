import re
import spacy
import torch
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k


def enable_gradients(model):
    for p in model.parameters():
        p.requires_grad = True


def disable_gradients(model):
    for p in model.parameters():
        p.requires_grad = False


def to_onehot(index, vocab_size):
    batch_size, seq_len = index.size(0), index.size(1)
    onehot = torch.FloatTensor(batch_size, seq_len, vocab_size).zero_()
    onehot.scatter_(2, index.data.cpu().unsqueeze(2), 1)
    return onehot


def load_dataset(batch_size):
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')
    url = re.compile('(<url>.*</url>)')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

    DE = Field(tokenize=list, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    EN = Field(tokenize=list, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    train, val, test = Multi30k.splits(exts=('.de', '.en'), fields=(DE, EN))
    DE.build_vocab(train.src)
    EN.build_vocab(train.trg)
    train_iter, val_iter, test_iter = BucketIterator.splits(
            (train, val, test), batch_size=batch_size, repeat=False)
    return train_iter, val_iter, test_iter, DE, EN
