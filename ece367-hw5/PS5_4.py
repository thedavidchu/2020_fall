"""
Code to validate

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


x1 = np.linspace(-1, 5, 121)
x2 = np.linspace(1, -1, 101)
x1, x2 = np.meshgrid(x1, x2)
X = np.stack((x1, x2))
X = np.transpose(X, axes=(1, 0, 2))

A = np.array([[0, 2], [1, 0], [2, 1]])
y = np.array([[-2, 5, 9]]).T

min_x = np.ones(48)
min_y = np.ones(48)

for γ in range(48):

    z = np.linalg.norm((A @ X - y), axis=1)**2 + γ * (np.abs(x1) + np.abs(x2))
    argmin = np.argmin(z)
    argmin = np.unravel_index(z.argmin(), z.shape)
    argmin = x1[0, argmin[1]], x2[argmin[0], 0]
    print(argmin)
    min_x[γ], min_y[γ] = argmin

    # Plot confusion shape - https://matplotlib.org/3.1.0/gallery/mplot3d/surface3d.html
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    # Set X-Y-Z labels
    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')
    # ax.set_zlabel('Penalty')

    # Plot
    # surf = ax.plot_surface(x1, x2, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()

plt.figure()
plt.plot(min_x, min_y, label='Minimum')
plt.show()