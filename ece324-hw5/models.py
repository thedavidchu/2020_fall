import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):

    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        # x = [sentence length, batch size]
        embedded = self.embedding(x)

        average = embedded.mean(0)  # [sentence length, batch size, embedding_dim]
        output = self.fc(average).squeeze(1)

        # Note - using the BCEWithLogitsLoss loss function
        # performs the sigmoid function *as well* as well as
        # the binary cross entropy loss computation
        # (these are combined for numerical stability)

        output = torch.sigmoid(output)

        return output


class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters=None, filter_sizes=None):
        super(CNN, self).__init__()

        # Group word vectors
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(2, embedding_dim))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(4, embedding_dim))

        # Pool along sentence
        self.pool = nn.AdaptiveMaxPool2d(1)

        # Fully Connected Layer
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        # Add word vector dimension
        embedded = self.embedding(x)  # [words in sentence (N), batch size (B), word vector(d=100)]
        N, B, d = embedded.shape
        embedded = torch.transpose(embedded, 0, 1)  # [B, N, d]
        embedded = embedded.view(B, 1, N, d)

        # Conv Layer 1
        a = self.conv1(embedded)
        a = torch.relu(a)
        a = self.pool(a)

        # Conv Layer 2
        b = torch.relu(self.conv2(embedded))
        b = self.pool(b)

        # Concatenate max-pooled conv
        x = torch.cat((a, b), dim=1)

        # Flatten x and MLP
        x = x.view(-1, 100)
        x = torch.sigmoid(self.fc(x))
        x = x.view(B)

        return x


class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim: int = 10):
        super(RNN, self).__init__()

        # Group word vectors
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)

        # Recurrent Neuron
        self.rec = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim)

    def forward(self, x, lengths=None):
        """
        Forward function, duh!
        :param x:
        :param lengths: tensor size(batch size) of lengths
        :return:
        """
        x = self.embedding(x)  # output: [words in longest sentence (N), sentences in batch (B), word vector size (d)]
        # print('length', lengths)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths)  # output: [sum words in batch (~N*B), d]

        # Run GRU on pack sequence
        _, x = self.rec(x)     # output: [sum words in batch (~N*B), d]
        # Strip off last value
        x = x[0, :, -1]
        x = torch.sigmoid(x)

        return x

