import scipy.io
import numpy as np


# Raw word vector
V = scipy.io.loadmat('wordVecV.mat')['V']

# Get titles
with open('wordVecTitles.txt', 'r') as f:
    titles = f.read().split('\n')

# Shape
m, n = V.shape

# Vector M (1 if >1, 0 if <1)
M = np.where(V > 0, 1, 0)
# Normalize M
M = M / np.linalg.norm(M, axis=0)


# ============================== PART C ============================== #
"""
(c) Use MATLAB svd command to compute the singular value decomposition of M˜ . List the 10
largest singular values in sorted order.
"""

U, s, VT = np.linalg.svd(a=M, full_matrices=False)
print('The ten singular values in increasing order are:\n', s)
S = np.diag(s)


# ============================== PART D ============================== #
"""
(d) In part (b) you assumed a low-rank approximation of M˜ and found an expression for the
document similarity. Let the distance between ith and jth documents be d(i, j) as per your
expression from part (b). Let the rank of your approximation be k where 0 < k ≤ min(m, n).
Compute d(i, j) for i, j ∈ [m] by assuming k = 9. Write down the titles of two most similar
documents.
"""

k = 9
min_arg = np.full(9+1, None)    # I added +1 because it's a pain to work with Matlab indices. Yes, it's not consistent.


def function(U, S, VT, k):
    """
    Find most similar distances by projecting onto k-th space and finding the distances between every vector in that space.
    :param U: U matrix
    :param S: S matrix - singular values
    :param VT: V.T matrix
    :param k: dimension to project onto
    :return: arguments of minimum
    """
    # Project each element onto subspace
    S_k = S[:k, :k]
    U_k = U[:, :k]
    VT_k = VT[:k, :]

    # Find M_k
    M_k = U_k @ S_k @ VT_k

    # Renormalize Columns
    M_k = M_k / np.linalg.norm(M_k, axis=0)
    # Store column dimension
    col = M_k.shape[1]

    # Find distances
    d = np.full((col, col), np.inf)
    for i in range(col):
        for j in range(col):
            if i == j:  # Unintentionally skipped half of array, but good!
                break
            d[i, j] = np.linalg.norm(M_k[:, i] - M_k[:, j])
    min = np.unravel_index(np.argmin(d), d.shape)

    return min


min_arg[9] = function(U, S, VT, k)
print(f'The articles that are closest for k = {k} are \"{titles[min_arg[k][0]]}\" and \"{titles[min_arg[k][1]]}\"')


# ============================== PART E ============================== #
"""
(e) Repeat what you did in part (d) with k = 8, 7, . . . , 1. What is the lowest k that does not
change your answer for part (d)? If your answer for lowest k is greater than 1 what is the
pair of most similar documents for k − 1?
"""

K = np.arange(8, 0, -1)
for k in K:
    min_arg[k] = function(U, S, VT, k)
    print(f'The articles that are closest for k = {k} are \"{titles[min_arg[k][0]]}\" and \"{titles[min_arg[k][1]]}\"')
