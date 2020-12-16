import numpy as np
import matplotlib.pyplot as plt

# ============================== PART A -- Calculate Gradients and Hessian ==============================

"""
Gradients and Hessians of functions.

i = 1:
    f(x, y) = 2x + 3y + 1

    grad(f_1(x, y)) = [2;
                       3]
    
    hess(f_1(x, y)) = [0, 0;
                       0, 0]

i = 2:
    f(x, y) = x^2 + y^2 − xy − 5

    grad(f_2(x, y)) = [2x-y;
                       2y-x]
    
    hess(f_2(x, y)) = [2, -1;
                       -1, 2]

i = 3:
    f(x, y) = (x−5)cos(y−5) − (y−5)sin(x−5)

    grad(f_3(x, y)) = [cos(y-5) - (y-5)cos(x-5);
                       -(x-5)sin(y-5) - sin(x-5)]
    
    hess(f_3(x, y)) = [(y-5)sin(x-5),          -sin(y-5) - cos(x-5);
                       -sin(y-5) - cos(x-5),   -(x-5)cos(y-5)]
"""

# ============================== PART B -- Plotting Level Sets ==============================

# Create grid to overlay
linspace = np.linspace(-2, 3.5, 100)
x, y = np.meshgrid(linspace, linspace)

p = np.zeros((100, 100, 2), dtype=np.float64)
p[:, :, 0] = x
p[:, :, 1] = y

# Define f_i for i = 1, 2, 3
f1 = lambda x, y: 2*x + 3*y + 1
f2 = lambda x, y: x**2 + y**2 - x*y - 5
f3 = lambda x, y: (x - 5) * np.cos(y - 5) - (y - 5) * np.sin(x - 5)

# Define gradients for f_i
grad_f1 = lambda x, y : np.array([2, 3], dtype=np.float64)
grad_f2 = lambda x, y : np.array([2 * x - y, 2 * y - x], dtype=np.float64)
grad_f3 = lambda x, y : np.array([np.cos(y - 5) - (y - 5) * np.cos(x - 5), -(x - 5) * np.sin(y - 5) - np.sin(x - 5)],
                                dtype=np.float64)
# Define Hessian for f_i
hess_f1 = lambda x, y : np.array([[0, 0], [0, 0]], dtype=np.float64)
hess_f2 = lambda x, y : np.array([[2, -1], [-1, 2]], dtype=np.float64)
hess_f3 = lambda x, y : np.array([[(y-5)*np.sin(x-5), -np.sin(y-5)-np.cos(x-5)], [-np.sin(y-5)-np.cos(x-5), -(x-5)*np.cos(y-5)]], dtype=np.float64)

# Calculate f_i
fig1 = f1(x, y)
fig2 = f2(x, y)
fig3 = f3(x, y)


# Calculate gradients
def gradient_and_tangent(x0, grad_fnc):

    # Calculate gradient
    m = grad_fnc(*x0)
    m = m / np.linalg.norm(m)
    grad = np.array([x0, x0 + m])

    # Calculate tangent
    neg_rec = np.array([m[1], -m[0]])
    neg_rec = neg_rec / np.linalg.norm(neg_rec)
    tan = np.array([x0 - 0.5*neg_rec, x0 + 0.5*neg_rec])

    return grad, tan


x0 = np.array([1, 0])

grad1, tan1 = gradient_and_tangent(x0, grad_f1)
grad2, tan2 = gradient_and_tangent(x0, grad_f2)
grad3, tan3 = gradient_and_tangent(x0, grad_f3)


# Plot figures in 2D
def plot_2d(figure, grad, tan, x, y, title):
    """
    Plot a 2D contour map with gradient and tangent.
    :param fig:
    :param grad:
    :param tan:
    :param title:
    :param x:
    :param y:
    :return:
    """
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(x, y, figure)
    fig.colorbar(cp)
    plt.plot(grad[:, 0], grad[:, 1], color='red', label='Gradient')
    plt.plot(tan[:, 0], tan[:, 1], color='orange', label='Tangent')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


plot_2d(fig1, grad1, tan1, x, y, '2x + 3y + 1 at (1, 0)')
plot_2d(fig2, grad2, tan2, x, y, 'x^2 + y^2 − xy − 5 at (1, 0)')
plot_2d(fig3, grad3, tan3, x, y, '(x − 5) cos(y − 5) − (y − 5) sin(x − 5) at (1, 0)')


# Create quadratic approximation
def f_quad_approx(f, grad, hess, x0, y0, p):
    """

    :param f: function
    :param grad: gradient function
    :param hess: Hessian function
    :param x0: x-coord of where we are approximating near
    :param y0: y-coord of where we are approximating near
    :param p: field over which all this happens
    :return:
    """
    diff = (p - np.array([x0, y0]))
    const = f(x0, y0)
    lin = np.dot(diff, grad(x0, y0))
    a = np.dot(diff, hess(x0, y0))
    b = np.zeros((a.shape[0], a.shape[1]))
    for i in range(len(a)):
        for j in range(len(a[i])):
            b[i, j] = a[i, j, 0] * diff[i, j, 0] + a[i, j, 1] * diff[i, j, 1]
    quad = 0.5 * b

    return const + lin + quad


f1_quad_approx = f_quad_approx(f1, grad_f1, hess_f1, *x0, p)
f2_quad_approx = f_quad_approx(f2, grad_f2, hess_f2, *x0, p)
f3_quad_approx = f_quad_approx(f3, grad_f3, hess_f3, *x0, p)


# Create 3D plot
def plot_3d(figure, approx, x, y, title):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, figure, cmap='viridis', edgecolor='none')
    ax.plot_surface(x, y, approx, cmap='binary', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


plot_3d(fig1, f1_quad_approx, x, y, '2x + 3y + 1 -- Quad Approx at (1, 0)')
plot_3d(fig2, f2_quad_approx, x, y, 'x^2 + y^2 − xy − 5 -- Quad Approx at (1, 0)')
plot_3d(fig3, f3_quad_approx, x, y, '(x − 5) cos(y − 5) − (y − 5) sin(x − 5) -- Quad Approx at (1, 0)')


# ============================== PART C -- Repeat for Different Points ==============================

# Repeat for (x, y) = (-0.7, 2)
x0 = np.array([-0.7, 2])

grad1, tan1 = gradient_and_tangent(x0, grad_f1)
grad2, tan2 = gradient_and_tangent(x0, grad_f2)
grad3, tan3 = gradient_and_tangent(x0, grad_f3)

plot_2d(fig1, grad1, tan1, x, y, '2x + 3y + 1 at (-0.7, 2)')
plot_2d(fig2, grad2, tan2, x, y, 'x^2 + y^2 − xy − 5 at (-0.7, 2)')
plot_2d(fig3, grad3, tan3, x, y, '(x − 5) cos(y − 5) − (y − 5) sin(x − 5) at (-0.7, 2)')

f1_quad_approx = f_quad_approx(f1, grad_f1, hess_f1, *x0, p)
f2_quad_approx = f_quad_approx(f2, grad_f2, hess_f2, *x0, p)
f3_quad_approx = f_quad_approx(f3, grad_f3, hess_f3, *x0, p)

plot_3d(fig1, f1_quad_approx, x, y, '2x + 3y + 1 -- Quad Approx at (-0.7, 2)')
plot_3d(fig2, f2_quad_approx, x, y, 'x^2 + y^2 − xy − 5 -- Quad Approx at (-0.7, 2)')
plot_3d(fig3, f3_quad_approx, x, y, '(x − 5) cos(y − 5) − (y − 5) sin(x − 5) -- Quad Approx at (-0.7, 2)')


# Repeat for (x, y) = (2.5, -1)
x0 = np.array([2.5, -1])

grad1, tan1 = gradient_and_tangent(x0, grad_f1)
grad2, tan2 = gradient_and_tangent(x0, grad_f2)
grad3, tan3 = gradient_and_tangent(x0, grad_f3)

plot_2d(fig1, grad1, tan1, x, y, '2x + 3y + 1 at (2.5, -1)')
plot_2d(fig2, grad2, tan2, x, y, 'x^2 + y^2 − xy − 5 at (2.5, -1)')
plot_2d(fig3, grad3, tan3, x, y, '(x − 5) cos(y − 5) − (y − 5) sin(x − 5) at (2.5, -1)')

f1_quad_approx = f_quad_approx(f1, grad_f1, hess_f1, *x0, p)
f2_quad_approx = f_quad_approx(f2, grad_f2, hess_f2, *x0, p)
f3_quad_approx = f_quad_approx(f3, grad_f3, hess_f3, *x0, p)

plot_3d(fig1, f1_quad_approx, x, y, '2x + 3y + 1 -- Quad Approx at (2.5, -1)')
plot_3d(fig2, f2_quad_approx, x, y, 'x^2 + y^2 − xy − 5 -- Quad Approx at (2.5, -1)')
plot_3d(fig3, f3_quad_approx, x, y, '(x − 5) cos(y − 5) − (y − 5) sin(x − 5) -- Quad Approx at (2.5, -1)')

# ============================== PART D -- Discussion ==============================
"""
My approximations are entirely accurate for f_1 and f_2 because they can both be expressed as second-order equations of 
x and y. However, for f_3, my approximation tended to be more accurate near the point about which I am making the 
approximation and where the curvature matches that of a parabola centred at the point.

...continue...
"""
