import warnings
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

part_C = not True
part_D = not True
part_E = not 90         # False, 10, 90

# ============================== Opening Files ============================== #
# Open and retrieve variables M and H
mat = scipy.io.loadmat('PS06_dataSet/sparseCoding.mat')
M, H = mat.get('M'), mat.get('H')
M_s = H @ M @ H.T


# ============================== PART C ============================== #
def histogram(img, title, bins: int = 255, range: tuple = None):
    # Plot histogram
    plt.figure(title)
    plt.hist(img.flatten(), bins=bins, range=range)  # arguments are passed to np.histogram

    plt.title(f"Histogram of {title}")
    plt.xlabel('Value')
    plt.ylabel('Frequency\nNumber of Instances')

    plt.show()
    return


if part_C:
    # Plot Histograms
    histogram(M, 'M')
    histogram(M_s, 'M^~')
    histogram(M_s, 'M^~ zoomed', bins=200, range=(-100, 100))

    # Number of Non-zero
    M_zeros = np.sum(M != 0)
    M_s_zeros = np.sum(M_s != 0)
    M_s_approx = np.sum(np.logical_or(-0.1 > M_s, M_s > 0.1))

    print(f'Non-zero elements\nM: {M_zeros}\nM~: {M_s_zeros}\nM~ within 0.1: {M_s_approx}')

"""
Non-zero elements
M: 65536
M~: 65325
M~ within 0.1: 63745
"""


# ============================== PART D ============================== #
def compression(X, M):
    """
    Return compression factor.
    :param X: compressed version of image
    :param M: original image
    :return: scalar - compression factor
    """
    non_zero = np.sum(X != 0)
    return non_zero / M.size


def MSE(M, N):
    """
    Calculate the mean-squared error between matrices
    :param M: Matrix 1
    :param N: Matrix 2
    :return: scalar - mean-squared error
    """
    if M.shape != N.shape:
        warnings.warn(f'Mismatched shape: M is shape {M.shape}, N is shape {N.shape}')
        return None

    mse = 1 / M.size * np.sum((M - N) ** 2)

    return mse


if part_D:
    λ = 30

    X = np.where(np.abs(M_s) > λ, M_s - λ * np.sign(M_s), 0)

    # Compression factor
    cf = compression(X, M)

    # Calculate MSE
    M_30 = np.around(H.T @ X @ H).astype(int)
    mse = MSE(M, M_30)

    print(f'The compression factor is {cf} and the MSE is {mse}')

    # Plot Histogram
    histogram(X[X != 0], title=f'Compression λ = {λ}')
    histogram(X[X != 0], title=f'Zoomed and Compressed λ = {λ}', bins=200, range=(-100, 100))

    # Plot Compression and Original
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.canvas.set_window_title(f'Original versus Compression λ = {λ}')
    ax1.set_title('Original')
    ax1.imshow(M, 'gray')
    ax2.set_title(f'Compression\nλ = {λ}')
    ax2.imshow(M_30, 'gray')
    plt.show()

# ============================== PART E ============================== #
"""
Change λ = {10, 90}
"""
if part_E:
    if part_E == True:
        λ = 10
    else:
        λ = part_E

    X = np.where(np.abs(M_s) > λ, M_s - λ * np.sign(M_s), 0)

    # Compression factor
    cf = compression(X, M)

    # Calculate MSE
    M_30 = np.around(H.T @ X @ H).astype(int)       # M_30 since lambda was 30... inaccurate now
    mse = MSE(M, M_30)

    print(f'The compression factor is {cf} and the MSE is {mse}')

    # Plot Histogram
    histogram(X[X != 0], title=f'Compression λ = {λ}')
    histogram(X[X != 0], title=f'Zoomed and Compressed λ = {λ}', bins=200, range=(-100, 100))

    # Plot Compression and Original
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.canvas.set_window_title(f'Original versus Compression λ = {λ}')
    ax1.set_title('Original')
    ax1.imshow(M, 'gray')
    ax2.set_title(f'Compression\nλ = {λ}')
    ax2.imshow(M_30, 'gray')
    plt.show()
