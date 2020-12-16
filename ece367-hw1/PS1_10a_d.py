import scipy.io
import numpy as np
import PS1_10e as pe

part_e = False
if part_e:
    f_term = pe.generate_frequency_vector()
else:
    mat = scipy.io.loadmat('wordVecV.mat')
    f_term = mat['V']

with open('wordVecTitles.txt') as f:
    titles = f.read().split('\n')

def print_titles(x):
    if isinstance(x, list):
        print('{')
        for i in x:
            print_titles(i)
        print('}')
    else:
        print(titles[x])

"""
|D| = 10
|W| = cardinality of W (max of 1651)

t belongs to |W|
d belongs to |D|

f_term(t, d)

"""

def euclidean_normalize(v):
    """
    Euclidean normalize a vector of dim 1
    :param v: 1-D np.array
    :return: normalized np.array
    """
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def euclidean_distance(array):
    """
    Find upper diagonal Euclidean distances.
    :param array:
    :return:
    """
    d = array.shape[0]
    distance = np.zeros(array.shape)
    for i in range(d):
        for j in range(i, d):
            distance[i, j] = np.linalg.norm(array[i] - array[j])
    return distance


def min_indices(array):
    """
    Searches upper diagonal (not including diagonal) for minimum value.
    :param array:
    :return:
    """
    min_index = [None, None]
    min = np.amax(array)
    for i in range(len(array)):
        for j in range(i+1, len(array)):
            if array[i, j] < min:
                min = array[i, j]
                min_index = [i, j]
    return min_index

# ============================== PART A  -- NEAREST DISTANCE AND ANGLE ==============================

v_d = f_term.T

def smallest_euclidean_distance(array):
    distance = euclidean_distance(array)
    min_dist_id = min_indices(distance)
    return min_dist_id

def smallest_angle(array):
    """
    Calculates smallest angle.
    Uses fact that the smallest distance and
    smallest angles are the same for vectors
    on a unit norm ball
    :param array:
    :return:
    """
    # Generate normalize array
    normed_array = np.zeros(array.shape)
    for i in range(len(array)):
        normed_array[i] = euclidean_normalize(array[i])

    # Calculate distance
    small_ang = smallest_euclidean_distance(normed_array)
    return small_ang

"""
Smallest Euclidean distance (indexing from zero): [6, 7]
    i.e. between {James Forder, Public image of George W. Bush}
Smallest angle (indexing from zero): [8, 9]
    i.e. between {Barack Obama, George W. Bush}
    
These are not the same pair. This may be due to the fact that the smallest distance is between two very 'short' 
vectors (i.e. not many words) so that they are close together since they are both close to the origin.
Meanwhile, the smallest angle relates the two articles that have a lot in common-- they were back-to-back POTUS.
"""


small_dist = smallest_euclidean_distance(v_d)
small_ang = smallest_angle(v_d)

print('\n============================== PART A==============================\n')
print('Smallest Euclidean distance (indexing from zero):', small_dist)
print_titles(small_dist)

print('Smallest angle (indexing from zero):', small_ang)
print_titles(small_ang)


# ============================== PART B -- NORMALIZED VECTORS ==============================

"""
Smallest Euclidean distance (indexing from zero): [8, 9]
    i.e. between {Barack Obama, George W. Bush}
Smallest angle (indexing from zero): [8, 9]
    i.e. between {Barack Obama, George W. Bush}

The answers are the same as the smallest angle in PART A. This makes sense since normalizing the vectors means that
absolute length no longer plays a bearing in determining how far away two vectors are (e.g. very long vectors with a
small angle between them may be farther apart than very short vectors with a large angle)

It thus makes sense that these two articles now align. Since we take the norm-1 of v_d, the large vectors are shrunk 
more than they would be by norm-2. This norm-1 means that many small word matches is linearly proportional to a single 
word matching many times. This helps broaden the scope of matches.
"""


sum = np.sum(f_term, axis=0)
v_tilde = v_d / np.sum(f_term, axis=0).reshape(10, 1)

tilde_dist = smallest_euclidean_distance(v_tilde)
tilde_ang = smallest_angle(v_tilde)


print('\n============================== PART B ==============================\n')
print('Smallest Euclidean distance (indexing from zero):', tilde_dist)
print_titles(tilde_dist)
print('Smallest angle (indexing from zero):', tilde_ang)
print_titles(tilde_ang)


# ============================== PART C ==============================
"""
Smallest Euclidean distance (indexing from zero): [8, 9]
    ie. between {Barack Obama, George W. Bush}
"""

f_doc = np.sum(f_term > 0, axis=1)
w = f_term / np.sum(f_term, axis=0) * np.sqrt(np.log(len(v_d)/f_doc)).reshape(1651, 1)

w_dist = smallest_euclidean_distance(w.T)
w_ang = smallest_angle(w.T)

print('\n============================== PART C ==============================\n')
print('Smallest Euclidean distance (indexing from zero):', w_dist)
print_titles(w_dist)


# ============================== PART D ==============================
print('\n============================== PART D ==============================\n')
"""
The reason for using the 'term frequency-inverse document frequency' (tf-idf) adjustment is that it weights the number 
of unique words shared between two documents more highly than sheer frequency. Words that are used more commonly in all 
documents will reduce the relevance of that word in determining how closely two articles match. This reduces the 
relevance of common words such as 'the' or 'a' or 'and'.

Geometrically, the tf-idf is finding a ratio between a word's frequency in an article and total words in the article. It
then multiplies that by a factor that decreases in magnitude as that word becomes more used across the other articles.
"""