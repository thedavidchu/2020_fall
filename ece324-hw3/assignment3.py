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
import scipy.signal
import matplotlib.pyplot as plt


try:
    from model import MultiLayerPerceptron
    from dataset import AdultDataset
    from util import *
except:
    print('Lol, gimme the files')
    from google.colab import files

    print('Select python files')
    uploaded = files.upload()
    from model import MultiLayerPerceptron
    from dataset import AdultDataset
    from util import *


"""
Adult income classification

In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

"""

"""
If you want to change the hyperparameters, check my main function. Sticking global variables at the top of a file is 
not a great idea for debugging, so I didn't. You're welcome. Everything is passed as a parameter into the main function.

By the way, these hyperparameters are just dummy hyperparamters (except for seed). I strongly disagree with the need for them just kicking 
around. Like why? Yes, so I used camelCaps so that it doesn't interfere with any real code. Thanks.

Sorry for the sass, but ain't nobody got time for poor coding.
"""
seed = 0

learningRate = 0.01
numLayers = 2
innerNeurons = 20
samplingFrequency = 10
lolFactor = 100


# =================================== Part 3.1 -- LOAD DATASET =========================================== #

######

# 3.1 YOUR CODE HERE

######

def load_data():
    try:
        data = pd.read_csv('adult.csv')
    except:
        uploaded = files.upload()
        data = {}
        for key in uploaded:
            file_obj = io.StringIO(str(uploaded[key])[2:-1].replace('\\n', '\n'))
            data[key] = pd.read_csv(key, delimiter=',')

        data = data['adult.csv']
    return data


data = load_data()

# =================================== Part 3.2 -- DATA VISUALIZATION =========================================== #

# the dataset is imported as a DataFrame object, for more information refer to
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# we can check the number of rows and columns in the dataset using the .shape field
# to get a taste of what our datset looks like, let's visualize the first 5 rows of the dataset using the .head() method
# the task will be to predict the "income" field (>50k or <50k) based on the other fields in the dataset
# check how balanced our dataset is using the .value_counts() method.

######

# 3.2 YOUR CODE HERE

######

print('========== Section 3.2 ==========')
print('Data:\n', data)
print('.shape:\n', data.shape)
print('.columns:\n', data.columns)
print('.head():\n', data.head())
print('["income"].value_counts():\n', data['income'].value_counts())

print(f'Proportion is: {11687 / 48842 * 100}% above 50k and {37155 / 48842 * 100}% below 50k')

"""
## Part 3.2 -- Understanding the Dataset

1. There are 11687 high income earners and 37155 low income earners.
2. This dataset is not as balanced as our first training dataset in Assignment 2 (ie. it is unbalanced). The high income earners make 24% (\~1/4) of the population, while the low income earners make up 76% (\~3/4).
"""


# =================================== DATA CLEANING =========================================== #

# datasets often come with missing or null values, this is an inherent limit of the data collecting process
# before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
# detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
# indicated with the symbol "?" in the dataset

# let's first count how many missing entries there are for each feature

######

# 3.3 YOUR CODE HERE

######

def missing_feature(data):
    col = data.columns
    for feature in col:
        r = data[feature].isin(["?"]).sum()
        print(f'There are {r} missing from {feature}')


missing_feature(data)


# col_names = data.columns
# print(col_names)
# num_rows = data.shape[0]
# print(num_rows)
# for feature in col_names:
#     #pass


#     r = data[feature].isin(["?"]).sum()
#     print(f'There are {r} missing from {feature}')

# next let's throw out all rows (samples) with 1 or more "?"
# Hint: take a look at what data[data["income"] != ">50K"] returns
# Hint: if data[field] do not contain strings then data["income"] != ">50K" will return an error

######

# 3.3 YOUR CODE HERE

#####

def remove_missing(data):
    for feature in data.columns:
        contains = data[feature].isin(["?"]).sum()
        if contains:
            data = data[data[feature] != '?']

    return data


# Keep raw data
# if True:
try:
    if raw_data.shape > data.shape:
        # print('Pass')
        pass
    elif data.shape > raw_data.shape:
        # print('Reset')
        raw_data = data.copy()
    else:
        # print('Error')
        pass
except:
    # print('Exception')
    raw_data = data.copy()  # This breaks if you run this script more than once!

print('========== Section 3.3 ==========')
# Clean data
data = remove_missing(data)
print(data.shape)

print(f'We removed {48842 - 45222} entries')
print(f'Thus we removed {3620 / 48842 * 100}% of the dataset')

"""## Part 3.3 -- Cleaning

1. We began with 48842 entries. We removed There are now 45222 entries. This means that 3620 were removed.
2. This is a reasonable number, because it makes up 7.5% of the dataset.
"""

# =================================== BALANCE DATASET =========================================== #

######

# 3.4 YOUR CODE HERE

######

rich = data[data['income'] == '>50K']
poor = data[data['income'] == '<=50K']

num_rich = rich.shape[0]
num_poor = poor.shape[0]

sample_num = min(num_rich, num_poor)

sample_rich = rich.sample(n=sample_num, random_state=seed)
sample_poor = poor.sample(n=sample_num, random_state=seed)

print('========== Section 3.4 ==========')
print('Shape of rich sample:', sample_rich.shape)
print('Shape of poor sample:', sample_poor.shape)

# =================================== DATA STATISTICS =========================================== #

# our dataset contains both continuous and categorical features. In order to understand our continuous features better,
# we can compute the distribution statistics (e.g. mean, variance) of the features using the .describe() method

######

# 3.5 YOUR CODE HERE

######

describe_rich = sample_rich.describe()
describe_poor = sample_poor.describe()
describe_data = data.describe()

print('========== Section 3.5 ==========')

for i in data.columns:
    freq = data.pivot_table(index=[i], aggfunc='size')
    print(freq)

verbose_print(describe_rich)
verbose_print(describe_poor)
verbose_print(describe_data)

"""## Part 3.5 -- Visualization and Understanding

1. The minimum age for the dataset is 17 and the minimum number of hours worked is 1.
"""

# likewise, let's try to understand the distribution of values for discrete features. More specifically, we can check
# each possible value of a categorical feature and how often it occurs
categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                     'relationship', 'gender', 'native-country', 'income']

for feature in categorical_feats:
    break   ### REMOVE!
    ######

    # 3.5 YOUR CODE HERE

    ######
    pie_chart(data, feature)

"""## Part 3.5 (cont'd)

2. Groups that are over/under-represented are:
>a. Engineers and other scientists are underrepresented in the Occupation feature.
>b. Gender. Females are very underrepresented (also non-binary persons are also not acknowledged in this dataset).
>c. Wives are also heavily underrepresented. In 1994, gay marriage was not legal in the USA, so all those husbands would have had to be married to women.

3. Other biases in the dataset include:
>a. Lumping together of separate races (e.g. Asian-Pacific Islander is one option)
>b. Gay marriage was only legalized in 2015. This means that gay men/ women were not considered married to each other and thus not a 'husband' or a 'wife'.
"""

# visualize the first 3 features using pie and bar graphs

######

# 3.5 YOUR CODE HERE

######

for feature in categorical_feats[:]:
    break                               ### REMOVE
    binary_bar_chart(data, feature)

"""## Part 3.5 (cont'd)

4. The top three features that distinguish between high and low income earners is:
>a. Education (Masters, Professional degrees, and Doctorates have more people making over 50k than under)
>b. Gender (men make more in this sample)
>c. Relationship status (married peoples tend to make more)

6. I would look at Education, Gender, and Relationship status. I would take the conditional probabilities of these seemingly 'most significant' factors.
"""

# =================================== DATA PREPROCESSING =========================================== #

# we need to represent our categorical features as 1-hot encodings
# we begin by converting the string values into integers using the LabelEncoder class
# next we convert the integer representations into 1-hot encodings using the OneHotEncoder class
# we don't want to convert 'income' into 1-hot so let's extract this field first
# we also need to preprocess the continuous features by normalizing against the feature mean and standard deviation
# don't forget to stitch continuous and cat features together

# ENCODE CATEGORICAL FEATURES

# Helpful Hint: .values converts the DataFrame to a numpy array

# LabelEncoder information: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
#       the LabelEncoder works by transforming the values in an input feature into a 0-to-"n_classes-1" digit label
#       if a feature in the data has string values "A, B, X, Y", then LabelEncoder will turn these into the numeric 0, 1, 2, 3
#       like other scikit-learn objects, the LabelEncoder must first fit() on the target feature data (does not return anything)
#       fitting on the target feature data creates the mapping between the string values and the numerical labels
#       after fitting, then transform() on a set of target feature data will return the numerical labels representing that data
#       the combined fit_transform() does this all in one step. Check the examples in the doc link above!

# label_encoder = LabelEncoder()
######

# 3.6 YOUR CODE HERE
labelencoder = LabelEncoder()
for feature in categorical_feats:
    data[feature] = labelencoder.fit_transform(data[feature])
    # print(label)
    pass  # replace with code that converts each string-valued feature into a numeric feature using the LabelEncoder

######

verbose_print(data)

# OneHotEncoder information: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
#    the OneHotEncoder works basically identical to the LabelEncoder
#    however, its input, instead of a single numeric array, is a matrix (dense or sparse) of 0 and 1 values
#    consider the following tabular data X of N data points (assume it is a data frame):
#
#    F1     F2      F3
#    ON     1     Toronto
#    ON     3     Scarborough
#    OFF    2     North York
#    ON     3     Toronto
#    OFF    3     Etobicoke
#    OFF    1     Scarborough
#    ...
#
#    F1 has 2 string values (ON, OFF), F2 has 2 numeric values (1, 2, 3), and F3 has 4 string values (Toronto, Scarborough,
#       North York, Etobicoke)
#    When we use the OneHotEncoder's fit_transform on this data frame X, the resulting matrix takes the shape: N x (2 + 3 + 4)
#
#    [[1 0 1 0 0 1 0 0 0]
#     [1 0 0 0 1 0 1 0 0]
#     [0 1 0 1 0 0 0 1 0]
#     [1 0 0 0 1 1 0 0 0]
#     [0 1 0 0 1 0 0 0 1]
#     [0 1 1 0 0 0 1 0 0]
#    ...
#
#    In other words, for tabular data with N data points and k features F1 ... Fk,
#    Then the resulting output matrix will be of size (N x (F1_n + ... + Fk_n))
#    This is because, looking at datapoint 2 for example: [1 0 0 0 1 0 1 0 0],
#    [1 0 | 0 0 1 | 0 1 0 0] -> here, [1 0] is the encoding for "ON" (ON vs OFF), [0 0 1] is the encoding for "3" (1 vs 2 vs 3), etc.
#    If a single _categorical variable_ F has values 0 ... N-1, then its 1-of-K encoding will be a vector of length F_n
#    where all entries are 0 except the value the data point takes for F at that point, which is 1.
#    Thus, for features F1 ... Fk, as described above, the length-Fi_n encodings are appended horizontally.

# firstly, we need to drop 'income' becaue we don't want to convert it into one-hot encoding:
if 'income' in data:
    print('income in data')
    y = data['income']
    data = data.drop(columns=['income'])
if 'income' in categorical_feats:
    categorical_feats.remove('income')
    y = y.values  # convert DataFrame to numpy array
# print(y)

# now, we can use the OneHotEncoder on the part of the data frame encompassed by 'categorial_feats'
# we can fit and transform as usual. Your final output one-hot matrix should be in the variable 'cat_onehot'
oneh_encoder = OneHotEncoder()
######

# 3.6 YOUR CODE HERE

######

cat_onehot = data[categorical_feats]
# verbose_print(cat_onehot)

bonus_categories = cat_onehot.copy()

cat_onehot = oneh_encoder.fit_transform(cat_onehot).toarray()
print(cat_onehot.shape)

# NORMALIZE CONTINUOUS FEATURES

# finally, we need to preprocess the continuous features by normalizing against the feature mean and standard deviation
# we begin by storing the data dropped of the categorical_feats in a separate variable, 'cts_data'
# your task is to use .mean() and .std() on this data to normalize it, then covert it into a numpy array

cts_data = data.drop(columns=categorical_feats).values
bonus_cts = cts_data.copy()

######

# 3.6 YOUR CODE HERE

######

def process_continuous(data):
    """
    :param data: np.array
    :return: (data - mean) / std
    """
    print(data.shape)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mean) / std


cts_data = process_continuous(cts_data)

print(type(cat_onehot))

# finally, we stitch continuous and categorical features
X = np.concatenate([cts_data, cat_onehot], axis=1)
bonus_X = np.concatenate([bonus_cts, bonus_categories], axis=1)
print(f'Shape of X = {X.shape}')
print(f"Shape of X_bonus = {bonus_X.shape}")

"""## Part 3.6 -- Pre-Processing

1. A disadvantage of using integers for categorical data is that it make the neural network think that two unrelated things are similar since the number is close together. For example, if 'Nurse' and 'Engineer' were consecutive numbers, the network would think that they are similar, when in fact they are quite different. Moreover, giving numerical weights to distinct categories implicitly adds an order to them, such as where 'Nurse' > 'Engineer' > 'Doctor'.

2. A disadvantage of using un-normalized, continuous data is that values with a large magnitudes will initially pull the neuron very strongly in that direction. Of course the neuron will learn to reduce the weight for this, but it will have to go through more training to figure this out.

NOTE: I didn't take out the index, so it got added in lol\

Bonus: (2 points) create a separate dataset where continuous features are un-normalized and
categorical features are represented as integers. Compare the performance of the neural network
on this dataset versus the one created as above. Report any differences in your report. Note that
the input size of the neural network will have to change with this different representation.
"""

# =================================== Part 3.7 -- MAKE THE TRAIN AND VAL SPLIT =========================================== #
# we'll make use of the train_test_split method to randomly divide our dataset into two portions
# control the relative sizes of the two splits using the test_size parameter

######

# 3.7 YOUR CODE HERE

######

X_train, X_test = train_test_split(X, test_size=0.2, random_state=seed)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=seed)

# Xb_train, Xb_test = train_test_split(bonus_X, test_size=0.2, random_state=seed)










# ... the void of dispair...



















# ...more void of despair...

















# ... almost there ...














# ... don't despair! ...






























X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()



# # Part 3.6 Bonus
# X_train = Xb_train
# X_test = Xb_test

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
    # model = main(lr=0.01, batch_size=64, plot_batch=True, smooth=False)  # Turn on or off smoothing!

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

