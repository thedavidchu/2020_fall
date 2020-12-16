# -*- coding: utf-8 -*-
"""assignment3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15LAmJrVwaBlAjPwsgxQgzzagw3wZqkgc

## Pandas from main.py
"""

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

"""
Adult income classification

In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

"""
seed = 0


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


import pickle

def vinegar(var, file):
    f = open(file, 'wb')
    pickle.dump(var, f)
    f.close()

def unpickle(file):
    with open(file, 'rb') as f:
        var = pickle.load(f)

    return var

# 5. According to the cleaned data, 2416/(2416+12367) of high school grads make over 50k. For Bachelors, 3178/(3178+4392) make over 50k. Thus, my best assumption given nothing else, is that they make less than 50k for both.

vinegar(X_train, 'X_train.pkl')
vinegar(X_test, 'X_test.pkl')
vinegar(y_train, 'y_train.pkl')
vinegar(y_test, 'y_test.pkl')

# vinegar(Xb_train, 'Xb_train.pkl')
# vinegar(Xb_test, 'Xb_test.pkl')