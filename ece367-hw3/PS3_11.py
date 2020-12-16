import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Load faces
img = scipy.io.loadmat('yalefaces.mat')['M']

# # Show first image
# plt.imshow(img[:,:,0]/255)

# Define constants
N = img.shape[-1]
d = 1024

# Find average X and reshape into 2D Matrix
X = img.reshape(d, N)
x_avg = X.mean(axis=1).reshape(d, 1)
X = X - x_avg

# Define C
C = X.dot(X.T)


# ============================== PART A ============================== #
"""
(a) In class you learned about singular value decomposition (SVD) and eigendecomposition.
What is the connection between the singular values of X and the eigenvalues of C? What
is the connection between the left-singular vectors of X and the eigenvectors of C? Make
sure to describe the reasoning behind your answers, e.g., by describing the singular or eigen
decompositions of each matrix.


The singular values of X are __ 

The left-singular vectors of X are the 
"""

# ============================== PART B ============================== #
print('=== PART B ===')
e_val, e_vect = np.linalg.eig(C)

print(f'Eigenvalue shape {e_val.shape}')
print(f'Eigenvector shape {e_vect.shape}')

# Sort in ascending order
order = e_val.argsort()[::-1]
e_val = e_val[order]
e_vect = e_vect[:, order]

# # Plot log(eigenvalue)
# plt.figure()
# plt.plot(np.log10(e_val), label='Eigenvalues')
# plt.title('Log(eigenvalues) by Magnitude in Descending Order')
# plt.xlabel('Eigenvalues')
# plt.ylabel('Order of Magnitude')
# plt.show()

"""
The eigenvalues are all real because C is a symmetric matrix.
"""

# ============================== PART C ============================== #
print('=== PART C ===')
e_face = e_vect.reshape(32, 32, d)

# fig = plt.figure()
# for i in range(10):
#     # Plot ten largest eigenvalues
#     fig.add_subplot(2, 5, i+1)
#     plt.title(f'{i+1}th Largest')
#     plt.imshow(e_face[:, :, i]/255)
#
# fig = plt.figure()
# for i in range(10):
#     # Plot ten smallest eigenvalues
#     fig.add_subplot(2, 5, i+1)      # row, col, position
#     plt.title(f'{i + 1}th Smallest')
#     plt.imshow(e_face[:, :, d-1-i]/255)

"""
The eigenvectors for the largest eigenvalues are clearly faces, while the eigenvectors for the smallest eigenvalues are not faces.

The reason for this difference, is because all the data are ('data' is plural, 'datum' is singular) composed of faces, which is a common attribute. The eigenvectors corresponding to the smallest eigenvalues represent less common features, hence they are small details (they sort of look like chins to me).
"""


# ============================== PART D ============================== #
print('=== PART D ===')


i_img = [1-1, 1076-1, 2043-1]       # Subtract 1 because MATLAB is dumb
j_dim = [2**(i+1) - 1 for i in range(10)]      # Subtract 1 to get identical results as Matlab version

def project(x, V):
    """
    Project vector x onto basis B
    :param x: column vector
    :param B: set of column vectors vectors in 2D matrix
    :return: y, column vector projection
    """

    # Number of dimensions to project onto
    # l = V.shape[1]

    # Iterate Is there are way to do this in NumPy?
    A = V.T @ V                     # A is an l x l matrix
    b = V.T @ x                     # b is an l size column vector

    # Solve Systems of Equation
    m = np.linalg.solve(A, b)       # m is an l size column vector
    y = V.dot(m)
    return y

# Run for each
Y = np.full((len(i_img), len(j_dim)), None)
for i, img in enumerate(i_img):
    for j, dim in enumerate(j_dim):
        ## Check shape etc
        B = e_vect[:, :dim+1]
        # Calculate projection and uncentre
        y = project(X[:, img:img+1], B)
        y = y + x_avg
        # Reshape and concatenate
        y = y.reshape(32, 32)
        Y[i, j] = y

# # Plot all 30 images
# fig = plt.figure()
# for i, img in enumerate(i_img):
#     for j, dim in enumerate(j_dim):
#         # Plot ten largest eigenvalues
#         fig.add_subplot(3, 10, (i)*10+(j+1))
#         plt.title(f'Img {img+1}\nProj {dim+1}')
#         plt.imshow(Y[i, j]/255)

# ============================== PART D ============================== #
print('=== PART E ===')

I1 = [1-1, 2-1, 7-1]   # Subtract 1 because Numpy vs Matlab
I2 = [2043-1, 2044-1, 2045-1]   # Subtract 1

B25 = e_vect[:, :25]
print(B25.shape)

# Calculate Projections
I = I1 + I2
C = np.full(len(I), None)
for i, img in enumerate(I):
    print(i)
    C[i] = project(X[:, img:img+1], B25)

# Calculate Euclidean Distance
dist = np.zeros((len(C), len(C)))
for i, ii in enumerate(C):
    for j, jj in enumerate(C):
        dist[i, j] = np.linalg.norm(ii - jj)


plt.figure()
plt.title('Matrix of Distances')
plt.imshow(dist, cmap='gray')
for i in range(len(C)):
    for j in range(len(C)):
        plt.text(x=i-0.5, y=j, s=str(round(dist[i, j])))

"""
The distances between the two people are stark; there is a clear gradient between the two -- the distance is much smaller between photos of the same person than it is of photos of the other person.

We can build a facial recognition model by measuring the Euclidean distance between a stored face vector and one that we are trying to recognize. The smaller the difference, the more likely a match.
"""