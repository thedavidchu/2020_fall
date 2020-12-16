import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Open MatLab file
mat = scipy.io.loadmat('pagerank_adj.mat')
J = mat['J'].astype(np.float64)
N = len(J)

with open('pagerank_urls.txt') as f:
    urls = np.array(f.read().split('\n'))

# Create link matrix
# [a b]
# [c d]
# A =
# [a/(a+b) b/(a+b)]
# [c/(c+d) d/(c+d)]

sum_row = np.sum(J, axis=0)
A = J / sum_row
x = np.ones((N, 1), dtype=np.float64)
k = np.arange(10)

# ============================== PART A -- Verify Columns of A ==============================
# Sum each column
a = np.sum(A, axis=0)  # ??
b = (np.logical_and(0.9999 < a, a < 1.0001))

# Check for non-ones
if False in b:
    print('Not all one!')
else:
    print('All values are 1 (plus or minus 0.0001)!')

"""
It is important that each column in A should sum to 1, because it represents the probability that one will randomly 
visit that link. Since you MUST both (a) visit a link and (b) visit a link on that page, we know that these numbers 
represent probabilities. Hence, we know that our total probability must be less than or equal to one, but we also know 
that we have the entire sample space, so it must be equal to 1.
"""


# ============================== PART B -- Eigenvectors ==============================

def error(A, x):
    dot = np.dot(A, x)
    r = np.linalg.norm(dot - x)
    return r


def log_plot(error, k, label='', title='Log Error vs Iteration'):
    log = np.log10(error)

    # plt.figure()
    plt.plot(k, log, label=f'{label} Error')
    plt.title(title)
    plt.xlabel('k')
    plt.ylabel('log10(error)')
    plt.legend()
    plt.show()


# Power Iteration (PI)
def power_iteration(A, x0, N: int = 10):
    x, y, l = [x0] + [None]*N, [None]*N, [None]*N

    for k in range(N):
        y[k] = A.dot(x[k])
        x[k+1] = y[k] / np.linalg.norm(y[k])
        l[k] = x[k + 1].T.dot(A).dot(x[k + 1])

    return x, y, l


pi_x, pi_y, pi_l = power_iteration(A, x, N=10)

pi_error = [error(A, pi_x[i]) for i in k]
log_plot(pi_error, k, 'Power Iteration')


# ============================== PART C -- Shift-Invert Power Iteration and Rayleigh ==============================


# # Get Pickle
# import pickle
#
#
# def vinegar(var, name):
#     f = open(f'{name}', 'wb')
#     pickle.dump(var, f)
#     f.close()
#
#
# def cucumber(name):
#     f = open(f'{name}', 'rb')
#     r = pickle.load(f)
#     f.close()
#     return r
#
#
# sip_x = cucumber('data_dump/sip_x.pkl')
# sip_y = cucumber('data_dump/sip_y.pkl')
# sip_l = cucumber('data_dump/sip_l.pkl')
#
# rq_x = cucumber('data_dump/rq_x.pkl')
# rq_y = cucumber('data_dump/rq_y.pkl')
# rq_sigma = cucumber('data_dump/rq_sigma.pkl')


# Shift-Invert Power Iteration (SIP)
def shift_invert_power(A, x0, sigma, N: int=10):
    x, y, l = [x0] + [None]*N, [None]*N, [None]*N

    eigen = A - sigma * np.eye(len(A))
    inv = np.linalg.inv(eigen)

    for k in range(N):
        y[k] = np.dot(inv, x[k])
        x[k+1] = y[k] / np.linalg.norm(y[k])
        l[k] = x[k + 1].T.dot(A).dot(x[k + 1])
    return x, y, l


sigma = 0.99
sip_x, sip_y, sip_l = shift_invert_power(A, x0=x, sigma=sigma, N=10)

sip_error = [error(A, sip_x[i + 1]) for i in k]
log_plot(sip_error, k, 'Shift-Invert Power Iteration')


# Rayleigh Quotient (RQ)
def rayleigh_quotient(A, x0, sigma1, sigma2, N: int = 10):
    x, y, sigma = [x0] + [None]*N, [None]*N, [sigma1, sigma2] + [None]*(N-2)

    for k in range(N):
        if k >= 2:
            sigma[k] = x[k].T.dot(A).dot(x[k]) / np.linalg.norm(x[k])**2    # Can we assume norm(x) == 1?

        eigen = A - sigma[k] * np.eye(len(A))
        try:
            y[k] = np.linalg.solve(eigen, x[k])
            # y[k] = np.linalg.inv(eigen) * x[k]    # This one breaks
            x[k+1] = y[k] / np.linalg.norm(y[k])
        except:
            print(f'Error on iteration {k}')
            return list(filter(None, x)), list(filter(None, y)), list(filter(None, sigma))

    return x, y, sigma


sigma1 = sigma2 = 0.99
rq_x, rq_y, rq_sigma = rayleigh_quotient(A, x, sigma1, sigma2, N=10)

rq_error = [error(A, rq_x[i + 1]) for i in range(len(rq_x) - 1)]
log_plot(rq_error, range(len(rq_x) - 1), 'Rayleigh Quotient')


# ============================== PART D -- PageRank Scores ==============================


def sort_pagerank(x):
    # Find page index and PageRank score
    a = list(enumerate(x))
    size = lambda pair: pair[1]
    a.sort(key=size)

    return a


extract_indices = lambda matrix: [i[0] for i in matrix]

# Sort based on PageRank score
a = sort_pagerank(pi_x[-1])
b = sort_pagerank(sip_x[-1])
c = sort_pagerank(rq_x[-1])

# Get top/bottom 5 of each
pi_bot5 = a[:5]
pi_top5 = a[-5:][::-1]

sip_bot5 = b[:5]
sip_top5 = b[-5:][::-1]

rq_bot5 = c[:5]
rq_top5 = c[-5:][::-1]

# Extract Indices
pi_bot5 = extract_indices(pi_bot5)
pi_top5 = extract_indices(pi_top5)

sip_bot5 = extract_indices(sip_bot5)
sip_top5 = extract_indices(sip_top5)

rq_bot5 = extract_indices(rq_bot5)
rq_top5 = extract_indices(rq_top5)


# Create bundles of all relevant data
def index_pagerank_url(indices, pagerank, urls):
    r = []
    for i in indices:
        r.append((i, pagerank[i][0], urls[i]))

    return r

pi_top5_url = index_pagerank_url(pi_top5, pi_x[-1], urls)
pi_bot5_url = index_pagerank_url(pi_bot5, pi_x[-1], urls)

sip_top5_url = index_pagerank_url(sip_top5, sip_x[-1], urls)
sip_bot5_url = index_pagerank_url(sip_bot5, sip_x[-1], urls)

rq_top5_url = index_pagerank_url(rq_top5, rq_x[-1], urls)
rq_bot5_url = index_pagerank_url(rq_bot5, rq_x[-1], urls)

def print_five(*args):
    l = ['PI', 'SIP', 'RQ']
    for i, x in enumerate(args):
        print(f'For {l[i]}:')
        for j, y in enumerate(x):
            print(f'{j+1}\t{y[0]}\t{y[1]}\t{y[2]}')
        print('\n')

print('Top 5')
print_five(pi_top5_url, sip_top5_url, rq_top5_url)


print('Bottom 5')
print_five(pi_bot5_url, sip_bot5_url, rq_bot5_url)

# # Store values
# vinegar(sip_x, 'sip_x.pkl')
# vinegar(sip_y, 'sip_y.pkl')
# vinegar(sip_l, 'sip_l.pkl')
#
# vinegar(rq_x, 'rq_x.pkl')
# vinegar(rq_y, 'rq_y.pkl')
# vinegar(rq_sigma, 'rq_sigma.pkl')