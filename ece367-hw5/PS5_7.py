"""
# Solve Quadratic Program for Investment Portfolio:

- Markowitz Portfolio

min p.T @ Σ @ p
pϵR^4

s.t.
    p.T * x_bar ⩾ r_min
    1.T * p = 1
    p ⩾ 0


- The solver requires this form:

min x.T @ P @ x
xϵR^4

s.t.
    Gx <= h
    Ax = b
"""

import numpy as np
import cvxopt as qp
import matplotlib.pyplot as plt

# Close previous graphs
plt.close('all')

# Define Given Variables
# Set up expected return and variance - p.T @ Σ @ p, where p is portfolio
x_bar = np.array([[1.1, 1.35, 1.25, 1.05]]).T
Σ = np.array([[0.2, -0.2, -0.12, 0.02],
              [-0.2, 1.4, 0.02, 0],
              [-0.12, 0.02, 1, -0.4],
              [0.02, 0, -0.4, 0.2]])

# Define Useful Variables
# Define steps of simulation
steps = 60
length = x_bar.shape[0]

# Define Variables to Record
r_min = np.linspace(1.05, 1.35, steps)
profit = np.zeros(steps)
variance = np.zeros(steps)
portfolio = np.zeros((steps, length))

for i, r in enumerate(r_min):
    """
    Find min variance to meet r_min constraint.
    Record the portfolio distribution and expected return.
    """
    # Define variables for the Quadratic Program
    # Quadratic Program x.T @ P @ x + q.T @ x
    P = qp.matrix(Σ)
    q = qp.matrix(np.zeros((length, 1)))
    # Inequality constraints Gx = h
    G = qp.matrix(-np.concatenate((x_bar.T, np.eye(length)), axis=0))
    h = qp.matrix(np.zeros((1 + length, 1)))
    h[0] = - r
    # Equality constraints Ax = b
    A = qp.matrix(np.ones((1, length)))
    b = qp.matrix(np.ones(1))

    # Solve the Quadratic Program
    # Solve QP and extract portfolio composition
    sol = qp.solvers.qp(P, q, G, h, A, b)
    x = np.array(sol['x'])

    # Record results
    # Record expected return, variance, and portfolio composition
    profit[i] = x_bar.T @ x
    variance[i] = x.T @ Σ @ x
    portfolio[i] = x[:, 0]

# ============================== PART A ============================== #
part_A = True
if part_A:
    # Plot Part A's graph of variance vs r_min
    plt.figure('r_min vs Variance')
    plt.title('r_min vs Variance')
    plt.plot(r_min, profit, 'r', label='Expected Return')
    plt.plot(r_min, variance, 'b', label='Variance')
    plt.legend()
    plt.xlabel('r_min')
    plt.ylabel('Variance')
    plt.show()

# ============================== PART B ============================== #
part_B = True
if part_B:
    # Plot Part B's portfolio distribution vs r_min
    plt.figure('Porfolio Composition')
    plt.title('Portfolio Composition')
    plt.plot(r_min, portfolio[:, 0], 'yellow', label='IBM')
    plt.plot(r_min, portfolio[:, 1], 'green', label='Google')
    plt.plot(r_min, portfolio[:, 2], 'red', label='Apple')
    plt.plot(r_min, portfolio[:, 3], 'blue', label='Intel. RIP my PEY')
    plt.legend()
    plt.xlabel('r_min')
    plt.ylabel('Composition')
    plt.show()
