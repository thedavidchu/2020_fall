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
#
# import storage
#
# storage.save(data_loader, '2_data_loader.pkl')


# ============================== Part 4 ============================== #
root = None
batch_size = 10
seed = 0
mean, std = normalize_data(root)
data_loader = import_data(root, None, mean, std)


import storage

storage.save(data_loader, '4_data_loader.pkl')