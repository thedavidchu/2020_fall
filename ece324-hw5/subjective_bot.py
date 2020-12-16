"""
    6. Testing on Your Own Sentence.

    You will write a python script that prompts the user for a sentence input on the command line, and prints the
    classification from each of the three models, as well as the probability that this sentence is subjective.

    An example console output:

        Enter a sentence
        What once seemed creepy now just seems campy

        Model baseline: subjective (0.964)
        Model rnn: subjective (0.999)
        Model cnn: subjective (1.000)

        Enter a sentence
"""

import torch
import torchtext
from torchtext import data
import spacy


# Create vocab object
def create_vocab(batch_size: int = 1):
    TEXT = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)
    train_data, val_data, test_data = data.TabularDataset.splits(
        path='./data/', train='train.tsv',
        validation='validation.tsv', test='test.tsv', format='tsv',
        skip_header=True, fields=[('text', TEXT), ('label', LABELS)])
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), batch_sizes=(batch_size, batch_size, batch_size),
        sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    TEXT.build_vocab(train_data, val_data, test_data)
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab

    return vocab


# Tokenizer
def tokenizer(text):
    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en(text)]


# Print Results Nicely
def print_results(a, b, c):
    det = []
    r = [a, b, c]
    for i, val in enumerate(r):
        if val >= 0.5:
            det.append('subjective')
        else:
            det.append('objective')

        if torch.is_tensor(val):
            r[i] = val.item()

    print('\nModel baseline: {} ({:.3f})'.format(det[0], r[0]))
    print('Model rnn: {} ({:.3f})'.format(det[1], r[1]))
    print('Model cnn: {} ({:.3f})\n'.format(det[2], r[2]))


# 1 - Obtain vocab object
vocab = create_vocab()

# 2 - Load models
baseline = torch.load('./model_baseline.pt')
cnn = torch.load('./model_cnn.pt')
rnn = torch.load('./model_rnn.pt')

while True:

    sentence = input('Enter a sentence\n')

    # 3 - Tokenize
    tokens = tokenizer(sentence)

    # 4 - Convert String token to int
    token_ints = [vocab.stoi[tok] for tok in tokens]

    # 5 - Convert list to LongTensor
    token_tensor = torch.LongTensor(token_ints).view(-1, 1)  # Shape is [sentence_len, 1]

    # 6 - Create tensor
    lengths = torch.Tensor([len(token_ints)])       # Called when running RNN model

    # Call Functions
    a = baseline(token_tensor)
    b = rnn(token_tensor, lengths)

    # Check if x needs padding for the CNN
    if token_tensor.shape[1] == 1:
        if token_tensor.shape[0] < 4:
            target = torch.zeros((4, 1), dtype=torch.long)
            target[:token_tensor.shape[0], :] = token_tensor
            token_tensor = target

    c = cnn(token_tensor)

    print_results(a, b, c)