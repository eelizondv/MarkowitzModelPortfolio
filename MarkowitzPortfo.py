# requirements
import numpy as np
import random
from qpsolvers import solve_qp
import matplotlib.pyplot as plt


## SETUP

# registation number
dig1 = 1
dig2 = 9
dummyrepetitions = 10 * dig1 + dig2

# parameters
n = 10

# generate data
for _ in range(dummyrepetitions):
    dummy = random.uniform(0, 1)

# create correlation matrix
Corr = np.array([[0] * n for _ in range(n)], dtype=float)
print(Corr)

for i in range(n):
    for j in range(n):
        Corr[i][j] = (-1) ** abs(i - j) / (abs(i - j) + 1)
print(Corr)

# initialize std dev and ER
ssigma = np.array([[0] * 1 for _ in range(n)], dtype = float)
mmu = np.array([[0] * 1 for _ in range(n)], dtype = float)

ssigma[0] = 2
mmu[0] = 3

print(ssigma)
print(mmu)

for i in range(n - 1):
    ssigma[i + 1] = ssigma[i] + 2 * random.uniform(0, 1)
    mmu[i + 1] = mmu[i] + 1

print(ssigma)
print(mmu)

# create covariance matrix
ddiag = np.array([[0] * n for _ in range(n)], dtype = float)
np.fill_diagonal(ddiag, ssigma)
print(ddiag)

C2 = np.matmul(np.matmul(ddiag,Corr), ddiag)
print(C2)

C = 0.5 * (C2 + C2.T)
print(C)

# ER vector
print(mmu)
mu = mmu.flatten()
print(mu)


## Task 1

# define r values
r_values = np.arange(2.00, 9.25, 0.25)
print(r_values)
print(r_values.size)

# t1
def t1(r_values):
    sigmas = []
    mus = []

    for r in r_values:
        # qp setup
        P = C
        q = np.zeros(n)
        G = -np.eye(n)
        h = np.zeros(n)
        A = np.vstack([mu, np.ones(n)])
        b = np.array([r, 1])

        # debug information
        print(f"Solving for r = {r}")
        print("P:", P)
        print("q:", q)
        print("G:", G)
        print("h:", h)
        print("A:", A)
        print("b:", b)

        # solve qp problem
        x = solve_qp(P, q, G, h, A, b, solver='quadprog')
        if x is None:
            print(f"No solution found for r = {r}")
            continue

        sigma = np.sqrt(np.dot(x.T, np.dot(C, x)))
        mu_val = np.dot(mu.T, x)

        sigmas.append(sigma)
        mus.append(mu_val)

    return sigmas, mus

# solve t1
sigmas, mus = t1(r_values)

# plot t1 results
plt.figure(figsize=(12, 8))
plt.scatter(sigmas, mus, c='black', marker='o', label='Portfolio')
plt.plot(sigmas, mus, linestyle='--', color='gray', alpha=0.5)

min_risk_idx = np.argmin(sigmas)
max_return_idx = np.argmax(mus)
plt.scatter(sigmas[min_risk_idx], mus[min_risk_idx], c='red', marker='x', s=100, label='Minimum Risk')
plt.scatter(sigmas[max_return_idx], mus[max_return_idx], c='green', marker='x', s=100, label='Maximum Return')

plt.xlabel('Risk (σ)', fontsize=14)
plt.ylabel('Return (μ)', fontsize=14)
plt.title('Efficient Frontier', fontsize=16)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend(fontsize=12)
plt.style.use('seaborn-v0_8-darkgrid')
plt.show()


# Task 2

# t2
def t2(r_values):
    sigmas = []
    mus = []

    for r in r_values:
        # qp setup
        P = C
        q = np.zeros(n)
        G = -np.eye(n)
        h = np.zeros(n)
        A = np.vstack([mu, np.ones(n)])
        b = np.array([r, 1])

        # debug information
        print(f"Solving for r = {r}")
        print("P:", P)
        print("q:", q)
        print("G:", G)
        print("h:", h)
        print("A:", A)
        print("b:", b)

        # modify the constraint
        G = np.vstack([G, np.ones(n)])
        h = np.hstack([h, [1]])

        # solve qp problem
        x = solve_qp(P, q, G, h, A[:-1], b[:-1], solver='quadprog')
        if x is None:
            print(f"No solution found for r = {r}")
            continue

        sigma = np.sqrt(np.dot(x.T, np.dot(C, x)))
        mu_val = np.dot(mu.T, x)

        sigmas.append(sigma)
        mus.append(mu_val)

    return sigmas, mus

# solve t2
sigmas_2, mus_2 = t2(r_values)

# plot t2 results
plt.figure(figsize=(12, 8))
plt.scatter(sigmas_2, mus_2, c='black', marker='o', label='Portfolio')
plt.plot(sigmas_2, mus_2, linestyle='--', color='gray', alpha=0.5)

min_risk_idx = np.argmin(sigmas_2)
max_return_idx = np.argmax(mus_2)
plt.scatter(sigmas_2[min_risk_idx], mus_2[min_risk_idx], c='red', marker='x', s=100, label='Minimum Risk')
plt.scatter(sigmas_2[max_return_idx], mus_2[max_return_idx], c='green', marker='x', s=100, label='Maximum Return')

plt.xlabel('Risk (σ)', fontsize=14)
plt.ylabel('Return (μ)', fontsize=14)
plt.title('Efficient Frontier', fontsize=16)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend(fontsize=12)
plt.style.use('seaborn-v0_8-darkgrid')
plt.show()


# Task 3

# t3
def t3(r_values):
    sigmas = []
    mus = []

    for r in r_values:
        # qp setup
        P = C
        q = np.zeros(n)
        G = -np.eye(n)
        h = np.zeros(n)
        A = mu.reshape(1, -1)
        b = np.array([r])

        # change equality constraint to inequality constraint
        G = np.vstack([G, -mu])
        h = np.hstack([h, -r])

        # solve qp problem
        x = solve_qp(P, q, G, h, A, b, solver='quadprog')
        if x is None:
            print(f"No solution found for r = {r}")
            continue

        sigma = np.sqrt(np.dot(x.T, np.dot(C, x)))
        mu_val = np.dot(mu.T, x)

        sigmas.append(sigma)
        mus.append(mu_val)

    return sigmas, mus

# solve t3
sigmas_3, mus_3 = t3(r_values)

# plot t3 results
plt.figure(figsize=(12, 8))
plt.scatter(sigmas_3, mus_3, c='black', marker='o', label='Portfolio')
plt.plot(sigmas_3, mus_3, linestyle='--', color='gray', alpha=0.5)

min_risk_idx = np.argmin(sigmas_3)
max_return_idx = np.argmax(mus_3)
plt.scatter(sigmas_3[min_risk_idx], mus_3[min_risk_idx], c='red', marker='x', s=100, label='Minimum Risk')
plt.scatter(sigmas_3[max_return_idx], mus_3[max_return_idx], c='green', marker='x', s=100, label='Maximum Return')

plt.xlabel('Risk (σ)', fontsize=14)
plt.ylabel('Return (μ)', fontsize=14)
plt.title('Efficient Frontier', fontsize=16)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend(fontsize=12)
plt.style.use('seaborn-v0_8-darkgrid')
plt.show()


# Task 4

# t4
def t4(r_values):
    sigmas = []
    mus = []

    for r in r_values:
        # qp setup
        P = C
        q = np.zeros(n)
        A = np.vstack([mu, np.ones(n)])
        b = np.array([r, 1])

        # remove non-negativity constraint
        G = None
        h = None

        # solve qp problem
        x = solve_qp(P, q, G, h, A, b, solver='quadprog')
        if x is None:
            print(f"No solution found for r = {r}")
            continue

        sigma = np.sqrt(np.dot(x.T, np.dot(C, x)))
        mu_val = np.dot(mu.T, x)

        sigmas.append(sigma)
        mus.append(mu_val)

    return sigmas, mus

# solve t4
sigmas_4, mus_4 = t4(r_values)

# plot t4 results
plt.figure(figsize=(12, 8))
plt.scatter(sigmas_4, mus_4, c='black', marker='o', label='Portfolio')
plt.plot(sigmas_4, mus_4, linestyle='--', color='gray', alpha=0.5)

min_risk_idx = np.argmin(sigmas_4)
max_return_idx = np.argmax(mus_4)
plt.scatter(sigmas_4[min_risk_idx], mus_4[min_risk_idx], c='red', marker='x', s=100, label='Minimum Risk')
plt.scatter(sigmas_4[max_return_idx], mus_4[max_return_idx], c='green', marker='x', s=100, label='Maximum Return')

plt.xlabel('Risk (σ)', fontsize=14)
plt.ylabel('Return (μ)', fontsize=14)
plt.title('Efficient Frontier', fontsize=16)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend(fontsize=12)
plt.style.use('seaborn-v0_8-darkgrid')
plt.show()