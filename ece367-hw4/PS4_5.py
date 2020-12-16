import numpy as np
import matplotlib.pyplot as plt


def check(p):
    # Check:
    all_position = np.arange(9, -1, -1) + 1/2
    all_velocity = np.ones(10)

    check_position = all_position @ p
    check_velocity = all_velocity @ p

    return check_position, check_velocity


def plot(p):
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

    plt.plot(pos, 'g-', label='Position')
    plt.plot(vel, 'y-', label='Velocity')
    plt.step(np.arange(0, len(p)), p, 'r-', label='Acceleration')
    plt.legend()
    plt.title(f'Position, Velocity, and Acceleration vs Time\nSum(p^2) = {np.linalg.norm(p)**2}')
    plt.xlabel('Time')
    plt.ylabel('Position, Velocity, Acceleration')

    print('Position vector', pos)
    print('Velocity vector', vel)

    return pos, vel, p


pow = lambda base, exponent: np.linalg.matrix_power(base, exponent)

A = np.array([[1, 1], [0, 1]])
X_0 = np.array([[0, 0]]).T
X_10 = np.array([[1, 0]]).T
b = np.array([[0.5, 1]]).T

# ============================== PART B ============================== #
part_b = True
if part_b:
    """
    X[10] = [pow(A, 9), pow(A, 8), pow(A, 7), pow(A, 6), pow(A, 5), pow(A, 4), pow(A, 3), pow(A, 2), pow(A, 1), pow(A, 0)] * b * [p1, p2, p3, ...].T *
    """
    # Create matrix A in y = A * x
    a = np.concatenate((pow(A, 9) @ b, pow(A, 8) @ b, pow(A, 7) @ b, pow(A, 6) @ b, pow(A, 5) @ b,
                        pow(A, 4) @ b, pow(A, 3) @ b, pow(A, 2) @ b, pow(A, 1) @ b, pow(A, 0) @ b), axis=1)

    # Solve with Moore-Penrose Inverse
    p = np.linalg.pinv(a) @ X_10

    # Check if the answer works
    check_pos, check_vel = check(p)
    pos, vel, p = plot(p)


# ============================== PART C ============================== #
part_c = False
if part_c:
    # Solve position at n = 5 (X[5] = [5; ?] = constraint_ans = constraint_row * p)
    zero = np.zeros((2, 1))
    constraint_row = np.concatenate((pow(A, 4) @ b, pow(A, 3) @ b, pow(A, 2) @ b, pow(A, 1) @ b, pow(A, 0) @ b,
                                     zero, zero, zero, zero, zero), axis=1)[0].reshape(1, -1)
    print(constraint_row)
    constraint_ans = np.zeros((1, 1))

    # Create standard matrix A in y = A * x
    a = np.concatenate((pow(A, 9) @ b, pow(A, 8) @ b, pow(A, 7) @ b, pow(A, 6) @ b, pow(A, 5) @ b,
                        pow(A, 4) @ b, pow(A, 3) @ b, pow(A, 2) @ b, pow(A, 1) @ b, pow(A, 0) @ b), axis=1)

    # Add extra constraints
    a = np.concatenate((a, constraint_row), axis=0)
    ans = np.concatenate((X_10, constraint_ans))

    # Solve with Moore-Penrose Inverse
    p = np.linalg.pinv(a) @ ans

    # Check if the answer works
    check_pos, check_vel = check(p)
    pos, vel, p = plot(p)