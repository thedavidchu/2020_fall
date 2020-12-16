import torch
import torch.nn as nn

import numpy as np
import io
import matplotlib.pyplot as plt

try:
    from google.colab import files
except:
    pass


class SingleNeuronClassifier(nn.Module):

    def __init__(self, activation_function=None, learning_rate=0.001, number_of_epochs=10000, random_seed=1):
        super(SingleNeuronClassifier, self).__init__()
        self.fc1 = nn.Linear(9, 1)

        # Hyperparameters
        if activation_function is None:
            self.fc1 = nn.Linear(9, 1)
        else:
            # Useable function
            self.ACT_FNC = activation_function

        self.LEARN_RATE = learning_rate
        self.EPOCHS = number_of_epochs
        self.R_SEED = random_seed

        self.loss_function = None
        self.optimizer = None

        # Training and validation data
        self.td, self.tl, self.vd, self.vl = self._load_data()

        # Plotting variables
        self.train_acc = []
        self.valid_acc = []
        self.train_loss = []
        self.valid_loss = []

    def forward(self, I):
        """
        From lecture.
        :param I:
        :return:
        """
        x = self.fc1(I)
        return x

    # ============================== INTERNAL FUNCTIONS ==============================

    def _random_init(self):
        """
        Seed the random generator.
        """
        torch.manual_seed(self.R_SEED)

    def _load_data(self):
        """
        Load the .csv data into the object
        """
        try:
            # Look for files
            train_data = np.loadtxt('traindata.csv', delimiter=',')
            train_label = np.loadtxt('trainlabel.csv', delimiter=',')

            valid_data = np.loadtxt('validdata.csv', delimiter=',')
            valid_label = np.loadtxt('validlabel.csv', delimiter=',')
        except:
            # Load files with google.colab
            uploaded = files.upload()

            data = {}
            for key in uploaded:
                file_obj = io.StringIO(str(uploaded[key])[2:-1].replace('\\n', '\n'))
                data[key] = np.loadtxt(key, delimiter=',')

            train_data = data['traindata.csv']
            train_label = data['trainlabel.csv']
            valid_data = data['validdata.csv']
            valid_label = data['validlabel.csv']

        train_data = torch.from_numpy(train_data).float()
        train_label = torch.from_numpy(train_label).float()
        valid_data = torch.from_numpy(valid_data).float()
        valid_label = torch.from_numpy(valid_label).float()

        return train_data, train_label, valid_data, valid_label

    def _accuracy(self, predictions, label):
        """
        Compute accuracy of predictions that have been made by comparing the predictions to a label.
        """
        correct = ((predictions.flatten() >= 0.5) == label.flatten()).float()
        mean = torch.mean(correct)
        return mean

    # ============================== RESET (HYPER)PARAMETERS ==============================

    def set_hyperparameters(self, activation_function=None, learning_rate=None, number_of_epochs=None,
                            random_seed=None):
        """
        Set hyperparameters.
        :param activation_function: function -- type of activation function
        :param learning_rate: scalar
        :param number_of_epochs: scalar -- number of training times
        :param random_seed: scalar -- seed for random number generator
        :return: None
        """

        if activation_function is not None:
            self.ACT_FNC = activation_function
        if learning_rate is not None:
            self.LEARN_RATE = learning_rate
        if number_of_epochs is not None:
            self.EPOCHS = number_of_epochs
        if random_seed is not None:
            self.R_SEED = random_seed

    # ============================== TRAIN ==============================

    def train_(self, epochs=None):
        """
        Trains for specified number of epochs.

        Referenced lecture material.
        """
        if epochs is None:
            epochs = self.EPOCHS

        # Set up function
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.LEARN_RATE)

        # Train for number of epochs
        for e in range(epochs):
            self.train_one_epoch()

    def train_one_epoch(self):
        """
        Source: lecture.
        :return:
        """
        self.optimizer.zero_grad()
        predict = self(self.td)
        train_loss = self.loss_function(input=predict.squeeze(), target=self.tl.float())
        train_loss.backward()
        self.optimizer.step()
        train_acc = self._accuracy(predict, self.tl)

        predict = self(self.vd)
        valid_acc = self._accuracy(predict, self.vl)
        valid_loss = self.loss_function(input=predict.squeeze(), target=self.vl.float())

        self.train_acc.append(train_acc)
        self.train_loss.append(train_loss)
        self.valid_acc.append(valid_acc)
        self.valid_loss.append(valid_loss)

    def plot(self):
        self.plot_loss()
        self.plot_accuracy()

    def plot_loss(self):
        """
        Plots training and validation loss against the training epoch.
        :return: None
        """

        title = ('Loss vs Epoch'
                 f'\nLinear Activation Function, '
                 f'Learning Rate of {self.LEARN_RATE}, '
                 f'Random Seed of {self.R_SEED}')

        # Plot loss
        plt.figure()
        plt.plot(self.train_loss, label='Training Loss')
        plt.plot(self.valid_loss, label='Validation Loss')
        plt.title(title)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        """
        Plots training and validation accuracy against the training epoch.
        :return: None
        """

        title = (f'Accuracy vs Epoch'
                 f'\nLinear Activation Function, '
                 f'Learning Rate of {self.LEARN_RATE}, '
                 f'Random Seed of {self.R_SEED}')

        # Plot accuracy
        plt.figure()
        plt.plot(self.train_acc, label='Training Accuracy')
        plt.plot(self.valid_acc, label='Validation Accuracy')
        plt.title(title)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    x = SingleNeuronClassifier(learning_rate=0.0001)
    x.train_(10000)
    x.plot()

    y = SingleNeuronClassifier(learning_rate=0.3)
    y.train_(10000)
    y.plot()

    z = SingleNeuronClassifier(learning_rate=0.1)
    z.train_(1000)
    z.plot()