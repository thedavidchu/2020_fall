import argparse
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import io
import matplotlib.pyplot as plt
import scipy.signal

try:
    from model import MultiLayerPerceptron
    from dataset import AdultDataset
    from util import *
except:
    from google.colab import files

    print('Select python files')
    uploaded = files.upload()
    from model import MultiLayerPerceptron
    from dataset import AdultDataset
    from util import *

# Import pickle

import pickle


def vinegar(var, file):
    f = open(file, 'wb')
    pickle.dump(var, f)
    f.close()


def unpickle(file):
    with open(file, 'rb') as f:
        var = pickle.load(f)

    return var


X_train = torch.from_numpy(unpickle('X_train.pkl')).float()
X_test = torch.from_numpy(unpickle('X_test.pkl')).float()
y_train = torch.from_numpy(unpickle('y_train.pkl')).float()
y_test = torch.from_numpy(unpickle('y_test.pkl')).float()


# # Part 3.6 Bonus
# X_train = torch.from_numpy(unpickle('Xb_train.pkl')).float()
# X_test = torch.from_numpy(unpickle('Xb_test.pkl')).float()

# =================================== Part 4.1 -- LOAD DATA AND MODEL =========================================== #

def load_data(batch_size: int = 64):
    ######

    # 4.1 YOUR CODE HERE

    ######

    train_loader = AdultDataset(X_train, y_train)  # , batch_size=batch_size, shuffle=False
    val_loader = AdultDataset(X_test, y_test)

    train_loader = DataLoader(train_loader, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_loader, batch_size=None, shuffle=False)

    return train_loader, val_loader


"""## Part 4.2 -- Data Loader

1. It is important to shuffle the data during training because otherwise the network will move in a predictable way every time. For example, if we were to have ordered the data by income, in one batch, the network would think that everything makes one more likely to be wealthier, then in the next, completely undoing this. This is like taking two steps forward and one step back.

## Part 4.3 -- Model

1. My first layer is 20 neurons. This is because there are 15 inputs there are in the unprocessed data. This means that we are looking for how 15 variables affect one's liklihood of making $50k. I increased it by 5 to represent the fact that we've added a lot more 'variables' since we've one-hot encoded them.

2. We think of the output as a probability because it will tend toward 1 when the input is more likely to make over \$50k, and it will tend toward 0 when they are likely to make under \$50k.
"""


def load_model(lr: int = 0.01, seed: int = 0, act_fnc: str = 'ReLU'):
    ######

    # 4.4 YOUR CODE HERE

    ######
    torch.manual_seed(seed=seed)
    model = MultiLayerPerceptron(input_size=104, act_fnc=act_fnc)

    # # Part 3.6 Bonus
    # model = MultiLayerPerceptron(input_size=14, inner_neurons=20)

    loss_fnc = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, loss_fnc, optimizer


def accuracy_and_correctness(predict, label):
    """
    Calculate accuracy and number of correct.
    :param predict: torch.tensor -- raw float predictions
    :param label: torch.tensor -- labels of data
    :return: accuracy, number of correct
    """
    correct = ((predict.flatten() > 0.5) == label.flatten()).float()
    mean = torch.mean(correct)
    total = torch.sum(correct)

    return mean, total


def avg_ten(lst):
    rec_avg = []
    for i in range(0, len(lst), 10):
        avg = sum(lst[i:i + 10]) / 10
        rec_avg.append(avg)

    return rec_avg


def every_ten(lst):
    rec = []
    for i in range(0, len(lst), 10):
        rec.append(lst[i])

    return rec


def plot_batch_(t_batch, t_loss, t_acc, v_batch, v_loss, v_acc, smooth=False, extra='', no_loss=False):
    """
    Plot loss and accuracy vs batch.
    :param t_batch:
    :param t_loss:
    :param t_acc:
    :param v_batch:
    :param v_loss:
    :param v_acc:
    :return:
    """
    if not no_loss:
        plt.figure()
        plt.plot(t_batch, t_loss, 'b:', label='Training')
        plt.plot(v_batch, v_loss, 'r-', label='Validation')
        if smooth:
            smooth_t_loss = scipy.signal.savgol_filter(t_loss, window_length=11, polyorder=5)
            plt.plot(t_batch, smooth_t_loss, 'g-', label='Smoothed Training')
        plt.title('Loss vs Batch' + extra)
        plt.legend()
        plt.xlabel('Batch')
        plt.ylabel('Loss')

    plt.figure()
    plt.plot(v_batch, v_acc, 'r-', label='Validation')
    plt.plot(t_batch, t_acc, 'b:', label='Training')
    if smooth:
        smooth_t_acc = scipy.signal.savgol_filter(t_acc, window_length=11, polyorder=5)
        plt.plot(t_batch, smooth_t_acc, 'g-', label='Smoothed Training')
    plt.title('Accuracy vs Batch' + extra)
    plt.legend()
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')


def plot_time_(t_time, t_loss, t_acc, v_time, v_loss, v_acc, smooth=False, extra='', no_loss=False):
    """
    Plot loss and accuracy vs time.
    :param t_time:
    :param t_loss:
    :param t_acc:
    :param v_time:
    :param v_loss:
    :param v_acc:
    :return:
    """

    t_time = np.array(t_time) - t_time[0]
    v_time = np.array(v_time) - v_time[0]

    if not no_loss:
        plt.figure()
        plt.plot(t_time, t_loss, 'b:', label='Training')
        plt.plot(v_time, v_loss, 'r-', label='Validation')
        if smooth:
            smooth_t_loss = scipy.signal.savgol_filter(t_loss, window_length=11, polyorder=5)
            plt.plot(t_time, smooth_t_loss, 'g-', label='Smoothed Training')
        plt.title('Loss vs Time' + extra)
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Loss')

    plt.figure()
    plt.plot(v_time, v_acc, 'r-', label='Validation')
    plt.plot(t_time, t_acc, 'b:', label='Training')
    if smooth:
        smooth_t_acc = scipy.signal.savgol_filter(t_acc, window_length=11, polyorder=5)
        plt.plot(t_time, smooth_t_acc, 'g-', label='Smoothed Training')
    plt.title('Accuracy vs Time' + extra)
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Accuracy')


def evaluate(model, val_loader, loss_fnc):
    """
    Evaluate model on validation data and label.
    :param model: model
    :param val_loader: validation data
    :param loss_fnc: loss function
    :return:
    """
    ######

    # 4.6 YOUR CODE HERE

    ######

    v_data = val_loader.dataset.X
    v_label = val_loader.dataset.y

    v_predict = model(v_data)
    v_loss = loss_fnc(input=v_predict.squeeze(), target=v_label)
    v_acc, v_cor = accuracy_and_correctness(v_predict, v_label)

    return v_acc, v_cor, v_loss


def main(lr: float = 0.01, batch_size: int = 64, epochs: int = 10, act_fnc='ReLU', eval_every=None, seed: int = 0,
         show=False, plot_batch=False, plot_time=False, smooth=None, no_loss=False):
    """
    I don't understand the fascination with global variables in the starter code. Honestly, it makes it harder to used.
    I'd appreciate if things were segmented into functions! I know that it's meant for a notebook, but still.

    :param lr:
    :param batch_size:
    :param epochs:
    :param act_fnc:
    :param eval_every:
    :param seed:
    :param show: show maximum accuracy
    :param plot_batch: plot accuracy/loss vs batch
    :param plot_time: plot accuracy/loss vs time
    :param smooth: smooth the training accuracy/loss?
    :param no_loss: don't display loss
    :return:
    """
    if smooth is not None:
        pass
    elif batch_size > 1000:
        smooth = False
    else:
        smooth = True

    if eval_every is not None:
        N0 = N1 = eval_every
    elif batch_size < 4:
        N0 = 100
        N1 = 100
    else:
        N0 = 1
        N1 = 10

    # Recording variables
    rec_t_batch, rec_t_acc, rec_t_cor, rec_t_loss = [], [], [], []
    rec_v_batch, rec_v_acc, rec_v_cor, rec_v_loss = [], [], [], []
    rec_t_time, rec_v_time = [], []

    # Load Neuron
    model, loss_fnc, optimizer = load_model(lr=lr, seed=seed, act_fnc=act_fnc)
    train_loader, val_loader = load_data(batch_size=batch_size)
    # Calculate max batch
    num_per_epoch = train_loader.dataset.X.shape[0]
    num_per_batch = train_loader.batch_size
    batch_per_epoch = int(num_per_epoch / num_per_batch)  # Round up if you don't chop off the end of a batch

    # if show:
    #     print('# For training:')
    #     print(f'Epoch\tBatch (of {batch_per_epoch})\tLoss\tNumber of Correct (of {batch_size})')

    # Training loop
    for e in range(epochs):
        for i, args in enumerate(train_loader):
            if i % N0 == 0:
                rec_t_time.append(time())

            # Train model
            t_data, t_label = args
            optimizer.zero_grad()
            t_predict = model(t_data)
            t_loss = loss_fnc(input=t_predict.squeeze(), target=t_label)
            t_loss.backward()
            optimizer.step()

            if i % N0 == 0:
                # Record Training data
                t_acc, t_cor = accuracy_and_correctness(t_predict, t_label)
                rec_t_batch.append(e * batch_per_epoch + i)
                rec_t_acc.append(t_acc)
                rec_t_cor.append(t_cor)
                rec_t_loss.append(t_loss)

            # Print Training Loss and Number of Correct
            # if show:
            #     if i % N1 == 0:
            #         print(f'{e}\t{i}\t{t_loss}\t{t_cor}')
            # Print (a) loss and (b) num correct every 10 batches
            if i % N1 == 0:
                # Record validation data
                rec_v_time.append(time())
                v_acc, v_cor, v_loss = evaluate(model, val_loader, loss_fnc)
                rec_v_batch.append(e * batch_per_epoch + i)
                rec_v_acc.append(v_acc)
                rec_v_cor.append(v_cor)
                rec_v_loss.append(v_loss)

    # Part 4.6: Validation -- Take every ten training points
    rec_t_batch = every_ten(rec_t_batch)
    rec_t_time = every_ten(rec_t_time)
    rec_t_loss, rec_t_acc = avg_ten(rec_t_loss), avg_ten(rec_t_acc)
    if show:
        print(f'Validation accuracy:\t{max(rec_v_acc)}')
    if plot_batch:
        plot_batch_(rec_t_batch, rec_t_loss, rec_t_acc, rec_v_batch, rec_v_loss, rec_v_acc, smooth=smooth,
                    extra=f'\nLearning Rate = {lr} Batch Size = {batch_size}', no_loss=no_loss)
    if plot_time:
        plot_time_(rec_t_time, rec_t_loss, rec_t_acc, rec_v_time, rec_v_loss, rec_v_acc, smooth=smooth,
                   extra=f'\nLearning Rate = {lr} Batch Size = {batch_size}', no_loss=no_loss)

    return model, rec_t_batch, rec_t_loss, rec_t_acc, rec_v_batch, rec_v_loss, rec_v_acc


if __name__ == "__main__":
    """
    Argparse is too much of a hassle to use. If you want something, change it with a parameter into the main function.
    
    I would like to point out, the reason why it is so messy to input is because of the structure of the starter code
    
    If the code were object oriented, we could store more information inside the object.
    
    For example, I could have functions to run every specific part.
    """
    # Part 4.6: Validation
    model = main(lr=0.01, batch_size=64, plot_batch=True, smooth=False)    # Turn on or off smoothing!

    # # Part 5.1: Learning Rate -- Vary lr between 1e-3 to 1e3
    # lr = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    # for i in lr:
    #     print(f'Learning Rate = {i}')
    #     main(lr=i, show=True)
    #
    # lr = [1e-2, 1, 1e2]
    # for i in lr:
    #     print(f'Learning Rate = {i}')
    #     main(lr=i, plot_batch=True)

    # # Part 5.3: Batch Size
    # main(batch_size=1, epochs=1, plot_time=True, plot_batch=True, show=True)
    # main(batch_size=64, epochs=10, plot_time=True, plot_batch=True, show=True)
    # main(batch_size=17932, epochs=50, plot_time=True, plot_batch=True)

    # Part 5.4: Under-fitting -- See model.py
    # main(lr=1, batch_size=64, plot_batch=True, show=True, no_loss=True)

    # Part 5.5: Over-fitting -- See model.py
    # main(lr=1, batch_size=64, plot_batch=True, show=True, no_loss=True)

    # Part 5.6: Activation Function -- See model.py
    # act_fnc = ['ReLU', 'tanh', 'sigmoid']
    # for i in act_fnc:
    #     t0 = time()
    #     model = main(lr=1, batch_size=64, plot_batch=True, show=True, act_fnc=i, no_loss=True)  # See model.py
    #     t1 = time()
    #
    #     print(f'Time elapsed: {t1 - t0}')

    pass

    # Part 3.6: Pre-Processing (BONUS)
    # main(lr=1, batch_size=64, plot_batch=True, show=True)
