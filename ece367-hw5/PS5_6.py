import numpy as np
import cvxopt
import matplotlib.pyplot as plt

# ============================== Instantiate Variables ============================== #

def instantiate():
    """
    X = array([[9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5, 0. ],
               [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 0. ]])

    X3 = array([[4.5, 3.5, 2.5, 1.5, 0.5, 0. , 0. , 0. , 0. , 0. ]])
    :return:
    """
    # Create A, b, p matrices
    A = np.array([[1, 1], [0, 1]])
    b = np.array([[0.5, 1]]).T

    pow = lambda base, exponent: np.linalg.matrix_power(base, exponent)

    X = np.concatenate((pow(A, 9) @ b, pow(A, 8) @ b, pow(A, 7) @ b,
                        pow(A, 6) @ b, pow(A, 5) @ b, pow(A, 4) @ b,
                        pow(A, 3) @ b, pow(A, 2) @ b, pow(A, 1) @ b,
                        pow(A, 0) @ b), axis=1)

    zero = np.zeros((2, 1))
    X3 = np.concatenate((pow(A, 4) @ b, pow(A, 3) @ b,
                         pow(A, 2) @ b, pow(A, 1) @ b,
                         pow(A, 0) @ b, zero, zero,
                         zero, zero, zero),
                        axis=1)[0].reshape(1, -1)

    return X, X3


def solve_lp(c: np.ndarray, G: np.ndarray, h: np.ndarray, A: np.ndarray = None, b: np.ndarray = None):
    """
    Solve LP given variables.

    :param c: np.ndarray
    :param G: np.ndarray
    :param h: np.ndarray
    :param A: np.ndarray
    :param b: np.ndarray
    :return: solution object
    """
    c_matrix = cvxopt.matrix(c)

    g_ineq = cvxopt.matrix(G)
    h_ineq = cvxopt.matrix(h)

    g_eq = cvxopt.matrix(A)
    h_eq = cvxopt.matrix(b)

    sol = cvxopt.solvers.lp(c=c_matrix, G=g_ineq, h=h_ineq, A=g_eq, b=h_eq)
    return sol


def plot(p, subtitle: str = '', text: bool = False):
    """
    Plot position, velocity, and acceleration/ force given a force for a unit mass.

    :param p: np.ndarray, (10x1) - acceleration/ force for unit mass
    :param text: bool - whether to display position and velocity vectors
    :return: position, velocity, and p vectors
    """
    # Instantiate x vector
    x = np.zeros((11, 2, 1))

    # Initialize initial conditions to 0
    x[0] = np.array([[0, 0]]).T

    # Create A, b, p matrices
    A = np.array([[1, 1], [0, 1]])
    b = np.array([[0.5, 1]]).T

    # Print minimized value
    print('Magnitude of sum(p^2) =', np.linalg.norm(p)**2)

    # Pad extra zero onto p
    p = np.concatenate((np.zeros((1, 1)), p), axis=0)

    # Iterate to find x through time
    for i in range(1, len(x)):
        x[i] = A @ x[i-1] + b * p[i]

    # Separate plots
    pos = x[:, 0, 0]
    vel = x[:, 1, 0]

    plt.figure(subtitle)
    plt.plot(pos, 'g-', label='Position')
    plt.plot(vel, 'y-', label='Velocity')
    plt.step(np.arange(0, len(p)), p, 'r-', label='Acceleration')
    plt.legend()
    plt.title(f'Position, Velocity, and Acceleration vs Time\n{subtitle}')
    plt.xlabel('Time')
    plt.ylabel('Position, Velocity, Acceleration')

    if text:
        print('Position vector', pos)
        print('Velocity vector', vel)

    return pos, vel, p


def print_force(p):
    p = np.around(p, decimals=4)
    for force in p:
        print(force[0])


# ============================== PART A ============================== #

def part_a(optional: bool = False):
    # ============================== Define Problem of l-1 ============================== #
    """
    min [0; 1].T @ [p; x]
    s.t.
        [0 -I] @ [p; x] <= [0; 0; 0]
        [I -I]
        [-I -I]
        
        [A 0] @ [p; x] = [1; 0]
        [A1 0] 
    """
    X, X3 = instantiate()

    I = np.eye(10)
    Z = np.zeros((10, 10))

    c = np.concatenate((np.zeros((10, 1)), np.ones((10, 1))), axis=0)

    G = np.block([[Z, -I], [I, -I], [-I, -I]])
    h = np.zeros((30, 1))

    if not optional:
        A = np.concatenate((X, np.zeros((2, 10))), axis=1)
        b = np.array([[1, 0]], dtype=float).T
    else:
        X = np.concatenate((X, X3), axis=0)
        A = np.concatenate((X, np.zeros((3, 10))), axis=1)
        b = np.array([[1, 0, 0]], dtype=float).T

    # ============================== Solve l-1 Problem ============================== #
    """
    Negatives due to Numeric instabilities.
    """
    sol = solve_lp(c, G, h, A, b)

    x = np.array(sol['x'])
    # s = np.array(sol['s'])
    # ans = G @ x + s           # = 0 = h
    p = x[:10]
    plot(p, subtitle=f'ℓ-1 Norm, Optional={optional}')
    print_force(p)

    return p

# ============================== PART B ============================== #


def part_b(optional: bool = False):
    # ============================== Define Problem of l-inf ============================== #
    """
    min [0; 1].T @ [p; x]
    s.t.
        [0 -I] @ [p; x] <= [0; 0; 0]
        [I -I]
        [-I -I]
        
        [A 0] @ [p; x] = [1; 0]
    """
    X, X3 = instantiate()

    I = np.eye(10)
    Zero1_10 = np.zeros((1, 10))
    One10_1 = np.ones((10, 1))

    c = np.concatenate((np.zeros((10, 1)), np.ones((1, 1))), axis=0)

    G = np.block([[Zero1_10, -1], [I, -One10_1], [-I, -One10_1]])
    h = np.zeros((21, 1))

    if not optional:
        A = np.concatenate((X, np.zeros((2, 1))), axis=1)
        b = np.array([[1, 0]], dtype=float).T
    else:
        X = np.concatenate((X, X3), axis=0)
        A = np.concatenate((X, np.zeros((3, 1))), axis=1)
        b = np.array([[1, 0, 0]], dtype=float).T

    # ============================== Solve l-inf Problem ============================== #
    sol = solve_lp(c, G, h, A, b)

    x = np.array(sol['x'])
    # s = np.array(sol['s'])
    # ans = G @ x + s  # = 0 = h
    p = x[:10]
    plot(p, subtitle=f'ℓ-∞ Norm, Optional={optional}')
    print_force(p)

    return p


part_a()
part_b()
part_a(True)
part_b(True)
