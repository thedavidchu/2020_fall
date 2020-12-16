import numpy as np
import random

"""
    3.1 Create train/validation/test splits

    This script will split the data/data.tsv into train/validation/test.tsv files.
"""

with open('data/data.tsv', 'r') as f:
    string = f.read().strip()

# Separate heading/data
entries = string.split('\n')
heading = entries[:1]
data = entries[1:]

# Get number of entries
N = len(data)

# Create Array to store
random.seed(0)
random_order = [None] * N

# Mix up entries
for i in range(N):
    index = random.randint(0, N - i - 1)
    random_order[i] = data.pop(index)

# Create training data
a = int(0.64 * N)
train = random_order[:a]

# Create validation data
b = int((0.64 + 0.16) * N)
valid = random_order[a:b]

# Create test data
test = random_order[b:]

# Create overfit data
overfit = random_order[:50]


def list_to_string(list, heading):
    string = heading[0]
    for i in range(len(list)):
        string += '\n' + list[i]

    string += '\n'
    return string


train = list_to_string(train, heading)
valid = list_to_string(valid, heading)
test = list_to_string(test, heading)
overfit = list_to_string(overfit, heading)


# Write string
with open('data/train.tsv', 'w') as f:
    f.write(train)

with open('data/validation.tsv', 'w') as f:
    f.write(valid)

with open('data/test.tsv', 'w') as f:
    f.write(test)

with open('data/overfit.tsv', 'w') as f:
    f.write(overfit)



#
# def read_data(file):
#     with open(file, 'r') as f:
#         string = f.read().strip()
#
#     raw = string.split('\n')
#     data = [raw[i].split('\t')[0] for i in range(1, len(raw))]
#     label = [int(raw[i].split('\t')[1]) for i in range(1, len(raw))]
#
#     return data, label
#
# # if __name__ == '__main__':
# #     # split_data()
# #     data, label = read_data('data/data.tsv')