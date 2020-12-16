import torch
import torch.optim as optim

import matplotlib.pyplot as plt
from time import time
import math

import torchtext
from torchtext import data
import spacy

import argparse
import os


from models import *

class TrainNN:
    def __init__(self, train_iter, valid_iter=None, test_iter=None,
                 embedding_dim=None, vocab=None, model=None,
                 optimizer=optim.Adam, lr=0.001, batch_size=64, epochs=25, seed=0,
                 loss_fnc=nn.BCEWithLogitsLoss
                 ):
        # Set Random Element
        self.seed = seed
        torch.manual_seed(self.seed)

        # Define Training
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.test_iter = test_iter

        # Store Hyper Parameters
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_fnc = loss_fnc

        # Declare Model for training
        if model is None:
            self.model = Baseline(embedding_dim=embedding_dim, vocab=vocab)
        else:
            self.model = model
        self.loss_fnc = loss_fnc()
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr)

        # Store training data
        self.max_epoch = 0
        self.max_v_acc = 0
        self.train_time = 0
        self.batch_per_epoch = math.ceil(len(self.train_iter.dataset) / self.batch_size)

        self.t_acc = []
        self.t_loss = []
        self.t_batch = []
        self.t_epoch = []

        self.v_acc = []
        self.v_loss = []
        self.v_batch = []
        self.v_epoch = []

    # Display Info
    def summary(self):
        print(f'Summary:\nMax Epoch\tMax v_acc\tTrain time\n{self.max_epoch}\t{self.max_v_acc}\t{self.train_time}')

    def plot(self, show_accuracy=True, show_loss=True, smooth=False, x='Epoch'):

        if x == 'Epoch':
            t_x = self.t_epoch
            v_x = self.v_epoch
        elif x == 'Gradient Step':
            t_x = [self.t_epoch[i] + self.t_batch[i] / self.batch_per_epoch for i in range(len(self.t_epoch))]
            v_x = [self.v_epoch[i] + self.v_batch[i] / self.batch_per_epoch for i in range(len(self.v_epoch))]
        elif x == 'Time':
            print('Time unimplemented')
            t_x = None
            v_x = None
        else:
            print('INVALID X COORDINATE')

        if show_accuracy:
            plt.figure()
            plt.plot(t_x, self.t_acc, 'b-', label='Training')
            plt.plot(v_x, self.v_acc, 'r-', label='Validation')
            plt.title(f'Accuracy vs. {x}\nBatchsize:{self.batch_size},LR:{self.lr},seed:{self.seed}')
            plt.legend()
            plt.xlabel(f'{x}')
            plt.ylabel('Accuracy')

        if show_loss:
            plt.figure()
            plt.plot(t_x, self.t_loss, 'b-', label='Training')
            plt.plot(v_x, self.v_loss, 'r-', label='Validation')
            plt.title(f'Loss vs. {x}\nBatchsize:{self.batch_size},LR:{self.lr},seed:{self.seed}')
            plt.legend()
            plt.xlabel(f'{x}')
            plt.ylabel('Loss')

    def test(self):
        avg_acc = 0
        avg_loss = 0
        batches_per_valid = len(self.test_iter.dataset) / self.test_iter.batch_size

        for i, iterate in enumerate(self.test_iter):
            v_data, v_len = iterate.text
            v_label = iterate.label

            v_predict = self.model(v_data, v_len)

            v_acc = self._accuracy(v_predict, v_label)
            v_loss = self.loss_fnc(input=v_predict, target=v_label.type_as(v_predict))

            avg_acc += v_acc / batches_per_valid
            avg_loss += v_loss.item() / batches_per_valid

        print(f'Test Set\nAccuracy\tLoss\n{avg_acc}\t{avg_loss}')

        return avg_acc, avg_loss

    # Train model
    def train(self, epochs=None, sample_batch: int = 1000, show=True):
        if epochs is None:
            epochs = self.epochs

        # Print Header
        print(f'epoch\tbatch\tt_acc\tt_loss\tv_acc\tv_loss')

        # Training loop
        for epoch in range(self.max_epoch, self.max_epoch + epochs):
            t0 = time()
            self._train_epoch(epoch, sample_batch=sample_batch, show=show)
            self.max_epoch += 1
            self.train_time += time() - t0

        print(f'\nTime for training so far: {self.train_time} s')

    # Internal Training Loop
    def _train_epoch(self, epoch, sample_batch: int = 1000, show=False):
        """
        Train for a single epoch for CNN/ baseline models.
        :param sample_epoch: int, testing var - sample every _ epochs
        :param sample_batch: int, testing var - sample every _ batches
        :param epoch: int - which epoch you are one
        :param show: bool - whether to show or not
        :return:
        """

        # Recorders
        t_acc_rec, t_loss_rec = [], []

        # Change sample_batch to something reasonable
        if sample_batch > self.batch_per_epoch:
            sample_batch = self.batch_per_epoch

        # Training batch loop
        for batch, iterate in enumerate(self.train_iter):
            t_data, t_len = iterate.text
            t_label = iterate.label

            # Gradient Descent
            self.optimizer.zero_grad()
            t_predict = self.model(t_data, t_len)
            t_loss = self.loss_fnc(input=t_predict, target=t_label.type_as(t_predict))  # Cast to same type
            t_loss.backward()
            self.optimizer.step()

            t_acc_rec.append(self._accuracy(t_predict, t_label))
            t_loss_rec.append(t_loss.item())

            if batch % sample_batch == sample_batch - 1:
                t_acc = self._record_training(epoch, batch, t_acc_rec, t_loss_rec)
                t_acc_rec, t_loss_rec = [], []

                if self.valid_iter is not None:
                    v_acc, v_loss = self._record_validation(epoch, batch)

                if show:
                    print(f'{epoch}\t{batch}\t{t_acc}\t{t_loss}\t{v_acc}\t{v_loss}')

    def _record_training(self, epoch, batch, t_acc_rec, t_loss_rec):
        """
        Record training accuracy.
        :param epoch:
        :param batch:
        :param t_acc_rec:
        :param t_loss:
        :return:
        """
        # Record training
        t_acc = sum(t_acc_rec) / len(t_acc_rec)
        t_loss = sum(t_loss_rec) / len(t_loss_rec)

        self.t_epoch.append(epoch)
        self.t_batch.append(batch)
        self.t_acc.append(t_acc)
        self.t_loss.append(t_loss)  # At specific point

        return t_acc

    def _record_validation(self, epoch, batch):
        """
        Perform validation loop. Why is there a batch size on the validation data?
        :param epoch: epoch number
        :param batch: batch number
        :return: None
        """
        avg_acc = 0
        avg_loss = 0
        batches_per_valid = len(self.valid_iter.dataset) / self.valid_iter.batch_size

        for i, iterate in enumerate(self.valid_iter):
            v_data, v_len = iterate.text
            v_label = iterate.label

            v_predict = self.model(v_data, v_len)
            v_acc = self._accuracy(v_predict, v_label)
            v_loss = self.loss_fnc(input=v_predict, target=v_label.type_as(v_predict))

            avg_acc += v_acc / batches_per_valid
            avg_loss += v_loss.item() / batches_per_valid

        self.v_epoch.append(epoch)
        self.v_batch.append(batch)
        self.v_acc.append(avg_acc)
        self.v_loss.append(avg_loss)

        # Maximum v_acc
        self.max_v_acc = max(self.max_v_acc, avg_acc)

        return avg_acc, avg_loss

    def _accuracy(self, prediction, labels):
        """
        Calculate accuracy and number of correct.
        :param prediction: torch.tensor -- raw float predictions
        :param labels: torch.tensor -- labels of data
        :return: accuracy
        """

        predict = (prediction > 0.5)

        # Calculate which ones are correct
        correct = (predict == labels).float()
        mean = torch.mean(correct)

        return mean


# ============================== OVERFIT ============================== #
def overfit_models(args, iter_type='bucket iterator'):
    """
    :return: info required to create model
    """
    TEXT = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)

    overfit_data, val_data, test_data = data.TabularDataset.splits(
        path='./data/', train='overfit.tsv',
        validation='validation.tsv', test='test.tsv', format='tsv',
        skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

    if iter_type == 'bucket iterator':
        overfit_iter, val_iter, test_iter = data.BucketIterator.splits(
            (overfit_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
            sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    else:
        overfit_iter, val_iter, test_iter = data.Iterator.splits(
            (overfit_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
            sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    TEXT.build_vocab(overfit_data, val_data, test_data)
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab
    embedding_dim = TEXT.vocab.vectors.shape[1]

    return overfit_iter, val_iter, test_iter, embedding_dim, vocab


def overfit_part_4(arguments):
    overfit_iter, val_iter, test_iter, embedding_dim, vocab = arguments

    base_line = TrainNN(overfit_iter, val_iter, test_iter, model=Baseline(embedding_dim=embedding_dim, vocab=vocab))
    base_line.train(1000)
    base_line.plot()
    base_line.test()

    return base_line


def overfit_part_5(arguments):
    overfit_iter, val_iter, test_iter, embedding_dim, vocab = arguments

    cnn = TrainNN(overfit_iter, val_iter, test_iter, model=CNN(embedding_dim=embedding_dim, vocab=vocab))
    cnn.train(50)
    cnn.plot()
    cnn.test()
    return cnn


def overfit_part_6(arguments):
    overfit_iter, val_iter, test_iter, embedding_dim, vocab = arguments

    rnn = TrainNN(overfit_iter, val_iter, test_iter, model=RNN(embedding_dim=embedding_dim, vocab=vocab))
    rnn.train(150)
    rnn.plot()
    rnn.test()
    return rnn

# ============================== MAIN ============================== #
def main(args, iter_type='bucket iterator'):
    ######
    # 3.2 Processing of the data
    # the code below assumes you have processed and split the data into
    # the three files, train.tsv, validation.tsv and test.tsv
    # and those files reside in the folder named "data".
    ######

    # 3.2.1
    TEXT = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)

    # 3.2.2
    train_data, val_data, test_data = data.TabularDataset.splits(
            path='./data/', train='train.tsv',
            validation='validation.tsv', test='test.tsv', format='tsv',
            skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

    # 3.2.3
    if iter_type == 'bucket iterator':
        train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
            sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    else:
        train_iter, val_iter, test_iter = data.Iterator.splits(
            (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
            sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    # 3.2.4
    TEXT.build_vocab(train_data, val_data, test_data)

    # 4.1
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab

    print("Shape of Vocab:", TEXT.vocab.vectors.shape)
    embedding_dim = TEXT.vocab.vectors.shape[1]

    return train_iter, val_iter, test_iter, embedding_dim, vocab


def part_4(args):
    train_iter, val_iter, test_iter, embedding_dim, vocab = args
    # ============================== PART 4 ============================== #

    # 4.3 - Baseline Model
    # See TrainNN class

    # 4.5 - Full Training
    a = TrainNN(train_iter, val_iter, test_iter, model=Baseline(embedding_dim=embedding_dim, vocab=vocab))
    a.train(25)
    a.plot()
    a.test()

    # 4.6 - Save Trained Model
    torch.save(a.model, 'model_baseline.pt')
    return a


def part_5(args):
    train_iter, val_iter, test_iter, embedding_dim, vocab = args
    # ============================== PART 5 ============================== #
    # 5.1 - Full Training
    a = TrainNN(train_iter, val_iter, test_iter, model=CNN(embedding_dim=embedding_dim, vocab=vocab))
    a.train(25)
    a.plot()
    a.test()

    # 5.1 - Save Trained Model
    torch.save(a.model, 'model_cnn.pt')
    return a


def part_6(args):
    train_iter, val_iter, test_iter, embedding_dim, vocab = args
    # ============================== PART 6 ============================== #

    a = TrainNN(train_iter, val_iter, test_iter, model=RNN(embedding_dim=embedding_dim, vocab=vocab))
    a.train(25)
    a.plot()
    a.test()

    # 5.1 - Save Trained Model
    torch.save(a.model, 'model_rnn.pt')
    return a


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='baseline',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)

    args = parser.parse_args()

    # ============================== Overfit ============================== #

    # arguments = overfit_models(args)

    # # 4.4 - Over-fitting to Debug
    # a = overfit_part_4(arguments)

    # # 5.1 - Overfit
    # a = overfit_part_5(arguments)

    # # 6.1 - Overfit
    # a = overfit_part_6(arguments)

    # ============================== Overfit ============================== #

    arguments = main(args, iter_type='bucket iterator')

    a = part_4(arguments)
    a = part_5(arguments)
    a = part_6(arguments)
