import numpy as np
import matplotlib.pyplot as plt

# ============================== PART A -- Calculate Gradients ==============================

"""
Gradients of functions.

i = 1:
    grad(f_1(x, y)) = [2, 3]^T

i = 2:
    grad(f_2(x, y)) = [2x-y, 2y-x]^T

i = 3:
    grad(f_3(x, y)) = [cos(y-5)-(y-5)cos(x-5), -(x-5)sin(y-5)-sin(x-5)]^T

"""

# ============================== PART B -- Plotting Level Sets ==============================
linspace = np.linspace(-2, 3.5, 100)
x, y = np.meshgrid(linspace, linspace)

p = np.zeros((100, 100, 2), dtype=np.float64)
p[:, :, 0] = x
p[:, :, 1] = y

# Define f_i for i = 1, 2, 3
f1 = lambda x, y: 2 * x + 3 * y + 1
f2 = lambda x, y: x * x + y * y - x * y - 5
f3 = lambda x, y: (x - 5) * np.cos(y - 5) - (y - 5) * np.sin(x - 5)

# Define gradients for f_i
grad_f1 = lambda x, y : np.array([2, 3], dtype=np.float64)
grad_f2 = lambda x, y : np.array([2 * x - y, 2 * y - x], dtype=np.float64)
grad_f3 = lambda x, y : np.array([np.cos(y - 5) - (y - 5) * np.cos(x - 5), -(x - 5) * np.sin(y - 5) - np.sin(x - 5)],
                                dtype=np.float64)
# Calculate f_i
fig1 = f1(x, y)
fig2 = f2(x, y)
fig3 = f3(x, y)

# Calculate gradients
x0 = np.array([1, 0])

m1 = grad_f1(1, 0)
m2 = grad_f2(1, 0)
m3 = grad_f3(1, 0)

grad1 = np.array([x0, x0 + m1])
grad2 = np.array([x0, x0 + m2])
grad3 = np.array([x0, x0 + m3])

# Calculate negative reciprocal
nr = lambda array : np.array([array[1], -array[0]])

nr1 = nr(m1)
nr2 = nr(m2)
nr3 = nr(m3)

tan1 = np.array([x0 - 0.5*nr1, x0 + 0.5*nr1])
tan2 = np.array([x0 - 0.5*nr2, x0 + 0.5*nr2])
tan3 = np.array([x0 - 0.5*nr3, x0 + 0.5*nr3])

# Plot contours
def plot_in_2d(fig1, fig2, fig3, x, y, approx=''):
    """
    Plot contours
    """
    # Figure 1
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(x, y, fig1)
    fig.colorbar(cp)
    plt.plot(grad1[:, 0], grad1[:, 1], label='Gradient')
    plt.plot(tan1[:, 0], tan1[:, 1], label='Tangent')
    plt.title(f'2x + 3y + 1{approx}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # Figure 2
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(x, y, fig2)
    fig.colorbar(cp)
    plt.plot(grad2[:, 0], grad2[:, 1], label='Gradient')
    plt.plot(tan2[:, 0], tan2[:, 1], label='Tangent')
    plt.title(f'x^2 + y^2 - xy - 5{approx}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # Figure 3
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(x, y, fig3)
    fig.colorbar(cp)
    plt.plot(grad3[:, 0], grad3[:, 1], label='Gradient')
    plt.plot(tan3[:, 0], tan3[:, 1], label='Tangent')
    plt.title(f'(x-5)*cos(y-5) - (y-5)*sin(x-5){approx}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


plot_in_2d(fig1, fig2, fig3, x, y)

# ============================== PART C -- Linear Approximation ==============================

def plot_approx_3d(fig1, approx1, fig2, approx2, fig3, approx3, x, y):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, fig1, cmap='viridis', edgecolor='none')
    ax.plot_surface(x, y, approx1, cmap='viridis', edgecolor='none')
    ax.set_title('2x + 3y + 1 -- Lin Approx at (1, 0)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, fig2, cmap='viridis', edgecolor='none')
    ax.plot_surface(x, y, approx2, cmap='viridis', edgecolor='none')
    ax.set_title('x^2 + y^2 - xy - 5 -- Lin Approx at (1, 0)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, fig3, cmap='viridis', edgecolor='none')
    ax.plot_surface(x, y, approx3, cmap='viridis', edgecolor='none')
    ax.set_title('(x-5)*cos(y-5) - (y-5)*sin(x-5) -- Lin Approx at (1, 0)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


f_approx = lambda f, grad, x0, y0: f(x0, y0) + np.dot((p - np.array([x0, y0])), grad(x0, y0))

f1_approx = f_approx(f1, grad_f1, 1, 0)
f2_approx = f_approx(f2, grad_f2, 1, 0)
f3_approx = f_approx(f3, grad_f3, 1, 0)

plot_approx_3d(fig1, f1_approx, fig2, f2_approx, fig3, f3_approx, x, y)
