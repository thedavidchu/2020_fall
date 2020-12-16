import argparse
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder           # Needed for torch?
from sklearn.model_selection import train_test_split

import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt


# ============================== Normalize Data ============================== #

def normalize_data(root):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    if root is None:
        data_set = torchvision.datasets.ImageFolder(root='./training_v2', transform=transform)
    else:
        data_set = torchvision.datasets.ImageFolder(root='./Chu_1004987311', transform=transform)

    data_loader = torch.utils.data.DataLoader(data_set, batch_size=len(data_set))

    # Find mean and std
    for i, batch in enumerate(data_loader):
        data, label = batch
        print(data.shape)
        mean = torch.mean(data, dim=(0, 2, 3)).numpy()
        std = torch.std(data, dim=(0, 2, 3)).numpy()
        print(f'Mean\t{mean}\tStd\t{std}')

    return mean, std


# ============================== Import Data ============================== #


def import_data(root, batch_size, mean, std):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)  # mean, std for all three channels
    ])

    if root is None:
        data_set = torchvision.datasets.ImageFolder(root='./training_v2', transform=transform)
    elif root == 'Part 2':
        data_set = torchvision.datasets.ImageFolder(root='./Chu_1004987311', transform=transform)
    else:
        data_set = torchvision.datasets.ImageFolder(root=root, transform=transform)
    if batch_size is None:
        batch_size = len(data_set)

    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)

    return data_loader


def show_img(img, mean, std, title):
    # Unnormalize
    img.shape
    img = img * std.reshape(3,1,1) + mean.reshape(3,1,1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


def show_batch(data_loader, mean, std):
    # Get first batch
    for batch in data_loader:
        data, label = batch
        break

    title = str([data_loader.dataset.classes[label[i]] for i in range(data_loader.batch_size)])[1:-1]\
        .replace('\'', '').replace(',', '')
    print(title)
    show_img(torchvision.utils.make_grid(data), mean, std, title)

    # for i in range(data_loader.batch_size):
    #     l = data_loader.dataset.classes[label[i]]
    #     print(f'{l}\t', end='')
    #
    # print('')


# ============================== Part 2 ============================== #
# root = 'Part 2'
# batch_size = 4
# mean, std = normalize_data(root)
# data_loader = import_data(root, batch_size, mean, std)
# show_batch(data_loader, mean, std)


# ============================== Part 4 ============================== #
root = None
batch_size = 10
seed = 0
mean, std = normalize_data(root)
data_loader = import_data(root, None, mean, std)








import argparse
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # Needed for torch?
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader

# import io
# import scipy.signal
import matplotlib.pyplot as plt
import torch.utils.data as data
import math
try:
    import storage  # Include with file or delete?
except:
    pass

class CNN_Part3(nn.Module):

    def __init__(self):
        super(CNN_Part3, self).__init__()

        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.input_mlp = 8 * 12 ** 2

        self.fc1 = nn.Linear(self.input_mlp, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, self.input_mlp)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class CNN(nn.Module):

    def __init__(self, num_conv: int = 2, num_kern: int = [4, 8], size_kern: int = [3, 5],
                 num_layers: int = 2, num_neurons=[100],
                 act_fnc=torch.relu, batch_norm=False):
        super(CNN, self).__init__()

        # Activation Function
        self.act_fnc = act_fnc
        self.batch_norm = batch_norm

        # Convolutions
        if isinstance(num_kern, (int, float)):
            num_kern = [int(num_kern)] * num_conv
        # No check if it is a list of the wrong size
        elif len(num_kern) != num_conv:
            print('INVALID ARRAY - CONV')

        if isinstance(size_kern, (int, float)):
            size_kern = [int(size_kern)] * num_conv
        elif len(size_kern) != num_conv:
            print('INVALID ARRAY - CONV (SIZE OF KERNEL)')

        num_kern = [3] + num_kern
        # print([nn.Conv2d(num_kern[i], num_kern[i + 1], size_kern[i]) for i in range(num_conv)])
        if batch_norm:
            self.conv_batch_norm = nn.ModuleList([nn.BatchNorm2d(num_kern[i + 1]) for i in range(num_conv)])
        self.conv = nn.ModuleList([nn.Conv2d(num_kern[i], num_kern[i + 1], size_kern[i]) for i in range(num_conv)])

        # Pool
        self.pool = nn.MaxPool2d(2, 2)

        # Determine how many MLP Neurons
        x = 56
        for i in range(num_conv):
            x = (x - (size_kern[i] - 1)) // 2
        self.input_mlp = num_kern[-1] * int(x) ** 2

        # MLP
        if isinstance(num_neurons, (float, int)):
            num_neurons = [int(num_neurons)] * (num_layers - 1)
        elif len(num_neurons) != num_layers - 1:
            print('INVALID ARRAY - MLP')

        num_neurons = [self.input_mlp] + num_neurons + [10]
        # print(num_neurons, num_layers)
        if batch_norm:
            self.mlp_batch_norm = nn.ModuleList([nn.BatchNorm1d(num_neurons[i + 1]) for i in range(num_layers - 1)])
        self.mlp = nn.ModuleList([nn.Linear(num_neurons[i], num_neurons[i + 1]) for i in range(num_layers)])

    def norm(self, x):
        """
        Normalize x. Adds small epsilon.
        :param x: data to normalize
        :return: normalized x
        """
        epsilon = 1e-5

        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)
        x = (x - mean) / (std + epsilon)
        return x

    def forward(self, x):
        """
        Forward function to evaluate network
        :param x:
        :return:
        """
        self.norm(x)

        # String
        for i, conv in enumerate(self.conv):
            x = conv(x)
            if self.batch_norm:
                x = self.conv_batch_norm[i](x)
            x = self.pool(self.act_fnc(x))

        x = x.view(-1, self.input_mlp)

        # Fully Connected Layer
        for i, mlp in enumerate(self.mlp[:-1]):
            x = mlp(x)
            if self.batch_norm:
                x = self.norm(x)
            x = self.act_fnc(x)

        # Last layer has no activation function nor batch norm
        x = self.mlp[-1](x)

        return x


class SignedDataset(data.Dataset):
    def __init__(self, data, label):
        super(SignedDataset, self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class train_CNN:
    def __init__(self,
                 data_loader=None, t_loader=None, v_loader=None,
                 model=CNN, num_conv: int = 2,
                 num_kern: list = [4, 8],
                 size_kern: list = 3,  # part 2[3, 5],
                 num_layers: int = 2,
                 num_neurons: int = 100,

                 # Training
                 loss_fnc=torch.nn.MSELoss,
                 optimizer=torch.optim.SGD,
                 act_fnc=torch.relu,

                 batch_norm: bool = False,

                 # Standard
                 epochs: int = 100,
                 batch_size: int = 64,
                 lr: float = 0.01,
                 seed=0
                 ):

        # Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

        # Data
        if data_loader is None:
            print('You are doing something wrong')
            self.t_loader = storage.load(f'4_t_loader_{batch_size}.pkl')
            self.v_loader = storage.load(f'4_v_loader_{batch_size}.pkl')
            # self.t_loader.batch_size = batch_size     # We disregard it!
            self.one_hot_label = True
        elif t_loader is None and v_loader is None:
            self.t_loader, self.v_loader = self._split(data_loader)
            self.one_hot_label = True

            storage.save(self.t_loader, f'4_t_loader_{batch_size}.pkl')
            storage.save(self.v_loader, f'4_v_loader_{batch_size}.pkl')
        else:
            self.t_loader = t_loader
            self.v_loader = v_loader
            self.one_hot_label = False

        # Cross Entropy Loss
        if loss_fnc == nn.CrossEntropyLoss:
            self.argmax_label = True
        else:
            self.argmax_label = False

        # GPU Stuff - All tensors must be on GPU
        # t_loader = t_loader.cuda()
        # v_loader = v_loader.cuda()

        # Model
        torch.manual_seed(self.seed)
        try:
            self.model = model(num_conv, num_kern, size_kern, num_layers, num_neurons, act_fnc, batch_norm)
        except:
            self.model = model()

        self.loss_fnc = loss_fnc()
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr)

        # Store training data
        self.max_epoch = 0
        self.max_v_acc = 0
        self.train_time = 0
        self.batch_per_epoch = math.ceil(len(self.t_loader.dataset) / self.batch_size)

        self.t_acc = []
        self.t_loss = []
        self.t_batch = []
        self.t_epoch = []

        self.v_acc = []
        self.v_loss = []
        self.v_batch = []
        self.v_epoch = []

    # ============================== Set Up Model ============================== #

    def _split(self, data_loader, test_size: int = 0.2):
        """
        Traing-Test split
        :param data_loader:
        :param test_size:
        :return:
        """
        for i, args in enumerate(data_loader):
            data, label = args

            t_data, v_data = train_test_split(data, test_size=test_size, random_state=self.seed)
            t_label, v_label = train_test_split(label, test_size=test_size, random_state=self.seed)

            t_dataset = SignedDataset(t_data, self._one_hot(t_label))
            v_dataset = SignedDataset(v_data, self._one_hot(v_label))

            t_loader = DataLoader(t_dataset, batch_size=self.batch_size, shuffle=True)
            v_loader = DataLoader(v_dataset, batch_size=None, shuffle=False)

        return t_loader, v_loader

    def cannabalize(self, a):
        """
        Eat an old train_CNN model and create a new one!

            - self.optimizer might not work...

        :param a:
        :return:
        """

        # Hyperparameters
        self.epochs = a.epochs
        self.batch_size = a.batch_size
        self.lr = a.lr
        self.seed = a.seed

        self.t_loader = a.t_loader
        self.v_loader = a.v_loader
        self.one_hot_label = a.one_hot_label

        # Model
        torch.manual_seed(self.seed)
        self.model = a.model
        # self.model = CNN_Part3()
        self.loss_fnc = a.loss_fnc
        # self.optimizer = a.optimizer(self.model.parameters(), lr=self.lr)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        # Store training data
        self.max_epoch = a.max_epoch
        self.max_v_acc = a.max_v_acc
        self.train_time = a.train_time
        self.batch_per_epoch = math.ceil(len(self.t_loader.dataset) / self.batch_size)

        self.t_acc = a.t_acc
        self.t_loss = a.t_loss
        self.t_batch = a.t_batch
        self.t_epoch = a.t_epoch

        self.v_acc = a.v_acc
        self.v_loss = a.v_loss
        self.v_batch = a.v_batch
        self.v_epoch = a.v_epoch

    # ============================== Measure Model ============================== #

    def summary(self):
        print(f'Max v acc {self.max_v_acc}\tMax epoch {self.max_epoch}\tTrain time{self.train_time}')

    def save(self, i):
        try:
            storage.save(self, f'4_a_{i}.pkl')
        except:
            print('Could not save!')

    def _accuracy(self, prediction, labels):
        """
        Calculate accuracy and number of correct.
        :param prediction: torch.tensor -- raw float predictions
        :param labels: torch.tensor -- labels of data
        :return: accuracy
        """

        predict = torch.argmax(prediction, dim=1)
        if self.argmax_label:
            label = labels  # Already argmax-ed # torch.argmax(labels, 1)
        else:
            label = torch.argmax(labels, dim=1)

        # print(predict.shape, label.shape)

        # Calculate which ones are correct
        correct = (predict == label).float()
        mean = torch.mean(correct)

        return mean

    def confusion_matrix(self):
        v_data, v_label = self.v_loader.dataset.data, self.v_loader.dataset.label
        v_predict = self.model(v_data)
        v_predict = torch.argmax(v_predict, dim=1)
        v_label = torch.argmax(v_predict, dim=1)
        r = confusion_matrix(v_label, v_predict)
        for i in r:
            for j in i:
                print(j, end='\t')
            print('')
        return r

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

    def _one_hot(self, labels):
        """
        One-hot encode a tensor of labels.

        ***Assume that you are encoding 0 to 9.

        :param labels: labels in range [0, 9]
        :return: one-hot encoded labels
        """
        switch = torch.Tensor(
            [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
        ).float()
        return switch[labels]

    # ============================== Train Model ============================== #

    def train(self, epochs=None, sample_batch: int = 1000, show=True):
        if epochs is None:
            epochs = self.epochs
        t0 = time()
        # Training loop
        for epoch in range(self.max_epoch, self.max_epoch + epochs):
            t0 = time()
            self.train_epoch(epoch, sample_batch=sample_batch, show=show)
            self.max_epoch += 1
            self.train_time += time() - t0
        self.max_epoch += 1  # Increment it to next
        # t1 = time()
        # self.train_time += t1 - t0
        print(f'\nTime for training loop: {self.train_time} s')

    def train_epoch(self, epoch, sample_batch: int = 1000, show=False):
        """
        Train for a single epoch.
        :param sample_epoch: int, testing var - sample every _ epochs
        :param sample_batch: int, testing var - sample every _ batches
        :param epoch: int - which epoch you are one
        :param show: bool - whether to show or not
        :return:
        """

        # Recorders
        t_acc_rec = []

        # Change sample_batch to something reasonable
        if sample_batch > self.batch_per_epoch:
            sample_batch = self.batch_per_epoch

        # Training batch loop
        for batch, args in enumerate(self.t_loader):
            t_data, t_label = args
            if self.one_hot_label and not self.argmax_label:
                pass
            elif not self.one_hot_label:
                t_label = self._one_hot(t_label)  # Is there a way to do all of this at the beginning?
            elif self.argmax_label:
                t_label = torch.argmax(t_label, dim=1)

            # Gradient Descent
            self.optimizer.zero_grad()
            t_predict = self.model(t_data)
            t_loss = self.loss_fnc(input=t_predict, target=t_label)
            t_loss.backward()
            self.optimizer.step()

            t_acc_rec.append(self._accuracy(t_predict, t_label))

            # if epoch % sample_epoch == 0:
            if batch % sample_batch == sample_batch - 1:
                # Record training
                t_acc = sum(t_acc_rec) / len(t_acc_rec)
                t_acc_rec = []
                self.t_epoch.append(epoch)
                self.t_batch.append(batch)
                self.t_acc.append(t_acc)
                self.t_loss.append(t_loss.item())

                if self.v_loader is not None:
                    v_data, v_label = self.v_loader.dataset.data, self.v_loader.dataset.label
                    v_predict = self.model(v_data)
                    if self.argmax_label:
                        v_label = torch.argmax(v_label, dim=1)
                    v_acc = self._accuracy(v_predict, v_label)
                    v_loss = self.loss_fnc(input=v_predict, target=v_label)

                    self.v_epoch.append(epoch)
                    self.v_batch.append(batch)
                    self.v_acc.append(v_acc)
                    self.v_loss.append(v_loss.item())

                    # Maximum v_acc
                    self.max_v_acc = max(self.max_v_acc, v_acc)

                if show:
                    print(f'{epoch}\t{batch}\t{t_loss}\t{t_acc}')


# ============================== Part 3 ============================== #
# Switch model to CNN_Part3!
# t_loader = data_loader
# a = train_CNN(data_loader=0, t_loader=t_loader, v_loader=None, model=CNN_Part3)
# a.train(show=True)
# a.plot(show_accuracy=True, show_loss=True)
#
# import torchsummary
#
# torchsummary.summary(a.model, (3, 56, 56))


# ============================== Part 4 ============================== #

# Set Up train_CNN Pickles
# a = train_CNN(data_loader=data_loader, batch_size=420)
# # a = train_CNN(data_loader=data_loader, batch_size=420)
# a.train()
# a.plot(show_accuracy=True, show_loss=True)


# Run specific case
# nc = [1, 2, 4]      # Number of Convolutional Layers
# nk = [10, 30]       # Number of Kernels per layer
# nnf = [8, 32]       # Number of Neurons in Fully connected
# lr = [0.1, 0.01]    # Learning Rate
# bs = [4, 32]        # Batch Size

# a = train_CNN(
#     data_loader=data_loader, epochs=110,
#     num_conv=nc[0], num_kern=nk[0], num_neurons=nnf[0], lr=lr[0], batch_size=bs[1]
# )
#
# param = [
#     [   1,  2,  4,  1,  1,  1,   1,     2,  2,  2,  2,  4],
#     [   10, 10, 10, 30, 10, 10,  10,    30, 10, 30, 30, 30],
#     [   8,  8,  8,  8,  32, 8,   8,     8,  32, 32, 32,  32],
#     [   0.1,0.1,0.1,0.1,0.1,0.01,0.1,   0.1,0.1,0.1,0.1,0.1],
#     [   32, 32, 32, 32, 32, 32,  4,     32, 32, 32, 4, 4]
# ]

# f = open('report.txt', 'a')
# report = 'Training Run\tMax V Acc\tTrain Time\tMax Epoch\tStatus\n'
# print(report)
# f.write(report)
# for i in range(12):
#     # Run farm
#     a = None
#     a = train_CNN(
#         data_loader=data_loader, epochs=110,
#         num_conv=param[0][i], num_kern=param[1][i], num_neurons=param[2][i], lr=param[3][i], batch_size=param[4][i]
#     )
#     try:
#         a.train()
#         print(f'{i}\t{a.max_v_acc}\t{a.train_time}\t{a.max_epoch}\tSuccess!\n')
#         report += f'{i}\t{a.max_v_acc}\t{a.train_time}\t{a.max_epoch}\tSuccess!\n'
#         f.write(f'{i}\t{a.max_v_acc}\t{a.train_time}\t{a.max_epoch}\tSuccess!\n')
#     except:
#         print(f'{i}\t{a.max_v_acc}\t{a.train_time}\t{a.max_epoch}\tFailure (probably memory or something)!\n')
#         report += f'{i}\t{a.max_v_acc}\t{a.train_time}\t{a.max_epoch}\tFailure (probably memory or something)!\n'
#         f.write(f'{i}\t{a.max_v_acc}\t{a.train_time}\t{a.max_epoch}\tFailure (probably memory or something)!\n')
#
#     # Summarize Parameters
#     import torchsummary
#     print(f'For number {i}')
#     torchsummary.summary(a.model, (3, 56, 56))
#     print('\n\n\n\n')


# f.close()


# ============================== PART 5 ============================== #

# a = train_CNN(
#     data_loader=data_loader, epochs=110,
#     num_conv=4, num_kern=30, num_neurons=32, lr=0.1, batch_size=4,
#     # batch_norm=True,
#     loss_fnc=nn.CrossEntropyLoss
# )
#
# import torchsummary
# torchsummary.summary(a.model, (3, 56, 56))
#
# report = 'CEL'
# print(report)
# a.train(10, sample_batch=100)
# try:
#     a.train(50)
#     print('Success\t', end='')
#     report += 'Success\t'
# except:
#     print('Failure!\t', end='')
#     report += 'Failure!\t'
#
# print(f'{a.max_v_acc}\t{a.train_time}\t{a.max_epoch}')
# report += f'{a.max_v_acc}\t{a.train_time}\t{a.max_epoch}\n'
# with open('report.txt', 'a') as f:
#     f.write(report)

# ============================== PART 6 ============================== #
"""
Best Model:
Epochs=50, Learning Rate=0.1, Batch size=4, seed=0
Convolutional layers=4, Kernels=30, kernel size=3x3, Neurons per layer=32, MLP Layers=2
Activation Function=ReLU, Loss function=Cross Entropy Loss, Batch norm=True, optimizer=SGD

Parameters=27,792, Memory=2.02 MB, Time=559 s
"""

"""
Best Small Model:
Epochs=97, Learning Rate=0.1, Batch size=4, seed=0
Convolutional layers=4, Kernels=10, kernel size=3x3, Neurons per layer=8, MLP layers=2
Activation function=ReLU, Loss function=MSE loss, Batch norm=False, optimizer=SGD

Parameters=3,188, Memory=0.4, Time=431 s
"""

# best_model = train_CNN(
#     data_loader=data_loader, epochs=110,
#     num_conv=4, num_kern=30, num_neurons=32, lr=0.1, batch_size=4,
#     batch_norm=True,
#     loss_fnc=nn.CrossEntropyLoss
# )
# best_model.train(50)
# best_model.confusion_matrix()
# torch.save(best_model.model.state_dict(), 'MyBest.pt')

# best_sm = train_CNN(
#     data_loader=data_loader, epochs=97,
#     num_conv=4, num_kern=10, num_neurons=8, lr=0.1, batch_size=4,
# )
# best_sm.train(97)
# best_sm.confusion_matrix()
# torch.save(best_sm.model.state_dict(), 'MyBestSmall.pt')
