import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image

try:
    from google.colab import files
except:
    pass


class SingleNeuronClassifier:

    def __init__(self, activation_function=None, learning_rate=0.001, number_of_epochs=10000, random_seed=1):
        """

        :param activation_function: string -- 'linear', 'sigmoid', or 'relu'
        :param learning_rate: scalar -- rate of learning
        :param number_of_epochs: scalar -- number of epochs it will train to by default
        :param random_seed: scalar -- default 1
        """
        # Hyperparameters
        if activation_function is None:
            # No input
            self.ACT_FNC = self.linear
        elif isinstance(activation_function, str):
            # String
            if activation_function.lower() == 'sigmoid':
                self.ACT_FNC = self.sigmoid
            elif activation_function.lower() == 'relu':
                self.ACT_FNC = self.relu
            else:
                self.ACT_FNC = self.linear
        else:
            self.ACT_FNC = self.linear

        self.LEARN_RATE = learning_rate
        self.EPOCHS = number_of_epochs
        self.R_SEED = random_seed

        # Parameters
        self.weights, self.bias = self._random_init()

        # Training and validation data
        self.td, self.tl, self.vd, self.vl = self._load_data()

        # Plotting variables
        self.train_acc = []
        self.valid_acc = []
        self.train_loss = []
        self.valid_loss = []

    def print_stuff(self, step=1000, specific=None):
        """
        Print accuracy and loss through epochs.
        :param step:
        :return:
        """
        print('Epoch\tTrain Acc\tValid Acc\tTrain Loss\tValid Loss')
        if specific is None:
            for i in range(0, len(self.train_acc), step):
                print(f'{i}\t{self.train_acc[i]}\t{self.valid_acc[i]}\t{self.train_loss[i]}\t{self.valid_loss[i]}')
        else:
            if isinstance(specific, (list, tuple)):
                for num in specific:
                    i = int(num)
                    print(f'{i}\t{self.train_acc[i]}\t{self.valid_acc[i]}\t{self.train_loss[i]}\t{self.valid_loss[i]}')
            elif isinstance(specific, (float, int)):
                i = int(specific)
                print(f'{i}\t{self.train_acc[i]}\t{self.valid_acc[i]}\t{self.train_loss[i]}\t{self.valid_loss[i]}')

        print('')

    def __call__(self, array=None):
        """
        Print current status of weights and bias. If an array is passed in, predict whether it is an X or not.
        :param array: array to be tested. np.array (9,)
        :return:
        """
        print('Weights:\n', self.weights)
        print('Bias:\n', self.bias)

        if array is not None:
            try:
                prediction = np.dot(self.weights, array) + self.bias[0]
                if prediction >= 0.5:
                    print('X')
                    return True
                else:
                    print('Not X')
                    return False
            except:
                pass

    # ============================== INTERNAL FUNCTIONS ==============================

    def _random_init(self):
        """
        Seed the random generator and create the random weights and bias.
        :returns:
            - np.array (9,) between (0, 1) -- random weights
            - np.array (1,) between (0, 1) -- random bias
        """
        np.random.seed(self.R_SEED)
        return np.random.random((9,)), np.random.random((1,))

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

        return train_data, train_label, valid_data, valid_label

    def _accuracy(self, prediction, label):
        """
        Calculate accuracy of prediction versus labels.
        :param prediction: np.array of predictions
        :param label: np.array of labels
        :return: accuracy as a decimal
        """
        correct_predict = (prediction == label)
        accuracy = np.average(correct_predict)
        return accuracy

    def _record_training(self, predictions, avg_loss):
        """
        Record the training accuracy and loss.
        :param predictions: np.array of boolean predictions for each entry (200,)
        :param avg_loss: scalar loss value
        :return: None
        """
        acc = self._accuracy(predictions, self.tl)
        self.train_acc.append(acc)
        self.train_loss.append(avg_loss)

    def _record_validation(self, predictions, avg_loss):
        """
        Record the validation accuracy and loss.
        :param predictions: np.array (20,) of booleans -- predict whether it is or is not an X
        :param avg_loss: scalar -- average loss
        :return: None
        """
        acc = self._accuracy(predictions, self.vl)
        self.valid_acc.append(acc)
        self.valid_loss.append(avg_loss)

    def _stringify_act_fnc(self):
        """
        Create a string out of the activation function.
        :return: String
        """
        if self.ACT_FNC == self.linear:
            return 'Linear'
        elif self.ACT_FNC == self.sigmoid:
            return 'Sigmoid'
        elif self.ACT_FNC == self.relu:
            return 'ReLU'
        else:
            return 'Unknown'

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

        if activation_function is None:
            # No input
            self.ACT_FNC = self.linear
        elif isinstance(activation_function, str):
            # String
            if activation_function.lower() == 'sigmoid':
                self.ACT_FNC = self.sigmoid
            elif activation_function.lower() == 'relu':
                self.ACT_FNC = self.relu
            else:
                self.ACT_FNC = self.linear
        else:
            # Useable function???
            self.ACT_FNC = activation_function
        if learning_rate is not None:
            self.LEARN_RATE = learning_rate
        if number_of_epochs is not None:
            self.EPOCHS = number_of_epochs
        if random_seed is not None:
            self.R_SEED = random_seed

    def set_perfect(self, k=1):
        self.weights = np.array([k, -k, k, -k, k, -k, k, -k, k])
        self.bias = np.array([0.5 - 5 * k])

    # ============================== TRAIN ==============================

    def train(self, epochs=None):
        """
        Trains for specified number of epochs.
        """
        if epochs is None:
            epochs = self.EPOCHS

        # Train for number of epochs
        for e in range(epochs):
            self.validate()
            self.train_one_epoch()

    def linear(self, z, validation=False):
        """
        Linear activation function.
        :param z: np.array -- result of the neuron's calculation.
        :param validation: boolean -- tells whether it is the validation data
        :return:
            - prediction: np.array of booleans -- predict whether X or not X
            - avg_loss: scalar -- average loss
            - dloss_db: np.array (1,) -- partial of loss wrt bias
            - dloss_dw: np.array (9,) -- partial of loss wrt weights
        """
        # Activation function
        Y = z
        # Calculate prediction
        prediction = (Y >= 0.5)

        if not validation:
            # Calculate difference
            diff = Y - self.tl
            # Loss gradients
            dloss_db = 2 * diff
            dloss_dw = dloss_db.reshape(200, 1) * self.td

            # Calculate loss
            avg_loss = np.average(diff * diff, axis=0)

            return prediction, avg_loss, dloss_db, dloss_dw

        else:
            # Calculate difference
            diff = Y - self.vl
            # Calculate loss
            avg_loss = np.average(diff * diff, axis=0)

            return prediction, avg_loss

    def sigmoid(self, z, validation=False):
        """
        Sigmoid activation function.
        :param z: np.array -- result of the neuron's calculation.
        :param validation: boolean -- tells whether it is the validation data
        :return:
            - prediction: np.array of booleans -- predict whether X or not X
            - avg_loss: scalar -- average loss
            - dloss_db: np.array (1,) -- partial of loss wrt bias
            - dloss_dw: np.array (9,) -- partial of loss wrt weights
        """

        # Activation function
        Y = 1 / (1 + np.exp(-z))
        # Calculate prediction
        prediction = (Y >= 0.5)

        if not validation:
            # Calculate difference
            diff = Y - self.tl
            # Loss gradients
            dloss_db = 2 * diff * np.exp(-z) * Y ** 2
            dloss_dw = dloss_db.reshape(200, 1) * self.td
            # Calculate loss
            avg_loss = np.average(diff * diff, axis=0)

            return prediction, avg_loss, dloss_db, dloss_dw
        else:
            # Calculate difference
            diff = Y - self.vl
            # Calculate loss
            avg_loss = np.average(diff * diff, axis=0)

            return prediction, avg_loss

    def relu(self, z, validation=False):
        """
        ReLU activation function.
        :param z: np.array -- result of the neuron's calculation.
        :param validation: boolean -- tells whether it is the validation data
        :return:
            - prediction: np.array of booleans -- predict whether X or not X
            - avg_loss: scalar -- average loss
            - dloss_db: np.array (1,) -- partial of loss wrt bias
            - dloss_dw: np.array (9,) -- partial of loss wrt weights
        """

        # Activation function
        Y = np.clip(z, 0, None)
        # Calculate prediction
        prediction = (Y >= 0.5)

        if not validation:
            # Calculate difference
            diff = Y - self.tl
            # Loss gradients
            dloss_db = np.where(z >= 0, 2 * diff, 0)
            dloss_dw = dloss_db.reshape(200, 1) * self.td
            # Calculate loss
            avg_loss = np.average(diff * diff, axis=0)

            return prediction, avg_loss, dloss_db, dloss_dw
        else:
            # Calculate difference
            diff = Y - self.vl
            # Calculate loss
            avg_loss = np.average(diff * diff, axis=0)

            return prediction, avg_loss

    def backprop(self, dloss_db, dloss_dw):
        """
        Backward propagate to adjust weights and bias.
        :param dloss_db: partial of loss wrt bias
        :param dloss_dw: partial of loss wrt weights
        :return: None
        """
        # Calculate avg(dloss/dw_i) and avg(dloss/db)
        avg_dloss_dw = np.average(dloss_dw, axis=0)
        avg_dloss_db = np.average(dloss_db)

        # Adjust w_i and b
        self.weights = self.weights - avg_dloss_dw * self.LEARN_RATE
        self.bias = self.bias - avg_dloss_db * self.LEARN_RATE

    def train_one_epoch(self):
        """
        Train model for one epoch.
        :param show: boolean indicating whether to print testing functions.
        """
        # Send input through neuron
        z = np.dot(self.td, self.weights) + self.bias
        # Activation function
        prediction, avg_loss, dloss_db, dloss_dw = self.ACT_FNC(z)
        # Record training data
        self._record_training(prediction, avg_loss)
        # # Calculate loss across all training examples
        self.backprop(dloss_db, dloss_dw)

    # ============================== TEST ==============================

    def validate(self):
        """
        Run the predictor against the validation data.
        :return: None
        """
        # Send input through neuron
        z = np.dot(self.vd, self.weights) + self.bias
        # Activation function
        prediction, avg_loss = self.ACT_FNC(z, validation=True)
        # Record validation data
        self._record_validation(prediction, avg_loss)

    # ============================== PLOT ==============================

    def plot(self):
        self.plot_loss()
        self.plot_accuracy()

    def plot_loss(self):
        """
        Plots training and validation loss against the training epoch.
        :return: None
        """

        title = ('Loss vs Epoch'
                 f'\n{self._stringify_act_fnc()} Activation Function, '
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
                 f'\n{self._stringify_act_fnc()} Activation Function, '
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

    def disp_kernel(self, kernel=None, ksize=3, isize=45):
        """
        Function to display, in gray scale, the weights in a grid
        the parameter 'kernel' contains ksize*ksize weights to display
        isize is the number of pixels on one dimentions of the square image to be displayed
        that is, the image to be displayed is isize * isize pixels
        NOTE THAT isize *must be divisable* by ksize
        :param kernel: array
        :param ksize: side length of grid
        :param isize: number of pixels in grid (divisible by ksize!)
        :return: None
        """

        if kernel is None:
            kernel = self.weights

        # for normalizing
        kmax = max(kernel)
        kmin = min(kernel)
        spread = kmax - kmin
        # print("max,min",kmax,kmin)

        dsize = int(isize / ksize)
        # print("dsize:",dsize)

        a = np.full((isize, isize), 0.0)

        # loop through each element of kernel
        for i in range(ksize):
            for j in range(ksize):
                # fill in the image for this kernel element
                basei = i * dsize
                basej = j * dsize
                for k in range(dsize):
                    for l in range(dsize):
                        a[basei + k][basej + l] = (kernel[(i * ksize) + j] - kmin) / spread

        # print(a)

        x = np.uint8(a * 255)

        # print(x)
        title = ('Kernel'
                  f'\n{self._stringify_act_fnc()} Activation Function, '
                  f'Learning Rate of {self.LEARN_RATE}, '
                  f'Random Seed of {self.R_SEED}')

        plt.figure()
        plt.title(title)
        img = Image.fromarray(x, mode='P')
        plt.imshow(img, cmap='Greys_r')
        plt.show()


# ============================== TESTING FUNCTIONS BELOW ==============================

# To test something, just type in the function's name into the console! Nothing happens by default.


def test_epochs():
    """
    Show accuracy as function of epoch.
    :return: x
    """
    x = SingleNeuronClassifier()
    x.train(5001)
    x.print_stuff(250)

    return x


def test_learning_rates():
    """
    Test learning rates.
    :return: None
    """
    lr = [1, 0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001, 0.0005, 0.0003, 0.0001]
    for i in range(len(lr)):
        print(f'For lr = {lr[i]}:')

        x = SingleNeuronClassifier('linear', learning_rate=lr[i], number_of_epochs=100)
        x.train()
        x.print_stuff(specific=50)


def test_activation_functions():
    """
    Compare activation functions.
    :return: None
    """
    x = SingleNeuronClassifier('linear', number_of_epochs=10001)
    y = SingleNeuronClassifier('sigmoid', number_of_epochs=10001)
    z = SingleNeuronClassifier('relu', number_of_epochs=10001)
    x.train()
    y.train()
    z.train()

    specific = [1e3, 5e3, 10e3]

    x.print_stuff(specific=specific)
    y.print_stuff(specific=specific)
    z.print_stuff(specific=specific)

    return x, y, z


def test_random_seed():
    """
    Test random seeds
    :return: None
    """
    for i in range(5):
        print(f'Seed = {i}')
        x = SingleNeuronClassifier('linear', number_of_epochs=1001, random_seed=i)
        x.train()
        x.print_stuff(specific=500)


def optimal_parameters():
    """
    Show optimal parameters
    :return: x
    """
    x = SingleNeuronClassifier(activation_function='relu', learning_rate=0.1, number_of_epochs=101, random_seed=2)
    x.train()
    x.print_stuff(10)
    x.train(1000)
    x.print_stuff(specific=1100)

    return x


def plot_slow_fast_medium():
    x = SingleNeuronClassifier(learning_rate=0.0001, number_of_epochs=1000)
    x.train()

    y = SingleNeuronClassifier(learning_rate=0.3, number_of_epochs=1000)
    y.train()

    z = SingleNeuronClassifier(learning_rate=0.1, number_of_epochs=1000)
    z.train()

    x.plot()
    x.disp_kernel()
    y.plot()
    y.disp_kernel()
    z.plot()
    z.disp_kernel()



if __name__ == '__main__':
    # Type in function names of what you want to test into the terminal!
    # Check if it has a return, and if it does, you can plot this!
    # e.g. x.plot() plots it. x.disp_kernel() displays the kernel.

    test_epochs()
    print('==============================')
    test_learning_rates()
    print('==============================')
    test_activation_functions()
    print('==============================')
    test_random_seed()
    print('==============================')
    optimal_parameters()
    print('==============================')
    plot_slow_fast_medium()