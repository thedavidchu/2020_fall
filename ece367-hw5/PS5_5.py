import numpy as np
import cvxopt


# Cost of corn, milk, bread
C = np.array([[0.15, 0.25, 0.05]]).T
# columns=vitamins, sugar, calories; rows=corn, milk, bread
D = np.array([[107, 45, 70], [500, 40, 121], [0, 60, 65]]).T

minimum = np.array([[5000, 0, 2000]]).T        # vitamins, sugar, calories
maximum = np.array([[10000, 1000, 2250]]).T     # vitamins, sugar, calories

"""
min c.T @ x
s.t. 
    2000 <= calories <= 2250
    5000 <= vitamins <= 10000
    sugar <= 1000
    ||x||.inf <= 10
"""
G = np.concatenate((D, -D[::2], np.eye(3), -np.eye(3)), axis=0)
H = np.concatenate((maximum, -minimum[::2], 10*np.ones((3,1)), np.zeros((3,1))), axis=0)

c = cvxopt.matrix(C)
g = cvxopt.matrix(G)
h = cvxopt.matrix(H)

sol = cvxopt.solvers.lp(c, g, h)
x = np.array(sol['x'])

print(f'x* = {x} and p* = {C.T.dot(x)}')

"""
x* = [corn = 6.58824399, milk = 9.99999983, bread = 5.0588161] and p* = $3.74117736.
 """