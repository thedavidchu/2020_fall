import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def inner_product(f, g):
    r = sp.integrate(f*g, (x, -sp.pi, sp.pi))
    return r


def norm(f):
    r = sp.sqrt(inner_product(f, f))
    return r


def project(x, V):
    r = inner_product(x, V) / inner_product(V, V) * V
    return r


def graham_schmidt(V, normalize=False):
    U = []
    for i in range(len(V)):
        u = V[i]
        for j in range(i):
            u = u - project(V[i], V[j])
        U.append(u)

    if normalize:
        for i in range(len(U)):
            U[i] = U[i] / norm(U[i])

    return U


def generate_matrix(x, V):
    l = len(V)
    A = sp.zeros(l, l)
    b = sp.zeros(l, 1)
    for i in range(l):
        for j in range(l):
            A[i, j] = inner_product(V[i], V[j])
        b[i] = inner_product(V[i], x)
    return A, b


def euclidean_length(V):
    s = 0
    for v in V:
        s += v**2
    return sp.sqrt(s)


# ============================== PART A ==============================
x, V, E, g = sp.symbols('x V E g')
g = sp.sin(x)

V = sp.Matrix([x**i for i in range(6)])
E = graham_schmidt(V, normalize=True)

# ============================== PART B ==============================
"""
a = [0, 0.987862135574674, 0, -0.155271410633429, 0, 0.00564311797634681]
"""

# Generate column of a values Ac = b
A, b = generate_matrix(g, V)
A_inv = A ** -1

# Solve for x
c = A_inv*b
c_eval = c.evalf()

# ============================== PART C -- Notice a Pattern ==============================
"""
The coefficients of a_0, a_2, and a_4 are all zero. This is intuitively correct, because these are even functions.
We know that the product of an even and an odd function is odd, and the integral of an odd function over the interval 
[-x0, x0] is equal to zero.
"""
# ============================== PART D ==============================
"""
The Taylor approximation is most accurate near x=0, but the error grows as abs(x) diverges from 0.
The projection, however, is much better in the end regions near x=-pi and x=+pi. Its error is fairly uniform throughout 
the region [-pi, +pi]. This is because these specific parameters best define the function sin(x) over this interval for 
polynomials with degree 5 or less. The projection approximation has smaller coefficients for the larger terms, which 
means that it does not diverge from sin(x) as quickly for larger numbers--however, near x=0, it is not as accurate as 
the Taylor approximation.
"""

# Create linspace line
N = 100
line = np.linspace(-np.pi, np.pi, N)
sin = np.sin(line)

# Figure out projection approximation
func = c.T * V
func = sp.lambdify(x, func)
g_proj = func(line).reshape(N,)

# Figure out Taylor's approximation
g_tay = line - np.power(line, 3) / 6 + np.power(line, 5) / 120

# Figure out error
proj_error = np.abs(sin - g_proj)
tay_error = np.abs(sin - g_tay)

# Plot
plt.subplot(1, 1, 1)
plt.plot(line, sin, label='sin(x)')
plt.plot(line, g_proj, label='Projection of sin(x)')
plt.plot(line, g_tay, label='x-x^2/6+x^5/120')
plt.plot(line, proj_error, label='Projection Error')
plt.plot(line, tay_error, label='Taylor Error')
plt.title('Projection of sin(x) vs Taylor Approx')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
