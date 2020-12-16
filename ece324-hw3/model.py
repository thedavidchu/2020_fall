import torch.nn as nn
import torch

class MultiLayerPerceptron(nn.Module):

    # ============================== Part 4.3 ============================== #

    def __init__(self, input_size, inner_neurons: int=20, act_fnc: str='ReLU'):

        super(MultiLayerPerceptron, self).__init__()

        self.fc1 = nn.Linear(input_size, inner_neurons)
        self.fc2 = nn.Linear(inner_neurons, 1)

        self.act_fnc = act_fnc

    # # Part 5.6
    def forward(self, features):
        if self.act_fnc == 'tanh':
            # print('tanh')
            features = torch.tanh(self.fc1(features))
        elif self.act_fnc == 'sigmoid':
            # print('sigmoid')
            features = torch.sigmoid(self.fc1(features))
        else:
            features = torch.relu(self.fc1(features))

        features = torch.sigmoid(self.fc2(features))
        return features

    # # ============================== Part 5.4: Under-fitting ============================== #
    #
    # def __init__(self, input_size, inner_neurons: int=0):
    #
    #     super(MultiLayerPerceptron, self).__init__()
    #
    #     self.fc1 = nn.Linear(input_size, 1)
    #
    # def forward(self, features):
    #     features = torch.sigmoid(self.fc1(features))
    #     return features

    # # ============================== Part 5.5: Over-fitting ============================== #
    #
    # def __init__(self, input_size, inner_neurons: int = 64):
    #     super(MultiLayerPerceptron, self).__init__()
    #
    #     self.fc1 = nn.Linear(input_size, inner_neurons)
    #     self.fc2 = nn.Linear(inner_neurons, inner_neurons)
    #     self.fc3 = nn.Linear(inner_neurons, inner_neurons)
    #
    #     self.fc4 = nn.Linear(inner_neurons, 1)
    #
    # def forward(self, features):
    #     features = torch.relu(self.fc1(features))
    #     features = torch.relu(self.fc2(features))
    #     features = torch.relu(self.fc3(features))
    #
    #     features = torch.sigmoid(self.fc4(features))
    #     return features