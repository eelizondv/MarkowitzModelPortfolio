## REQUIREMENTS

import numpy as np
import random
from qpsolvers import solve_qp
import matplotlib.pyplot as plt


## SETUP

# Replace with actual digits from registration number
dig1 = 0
dig2 = 0
dummyrepetitions = 10 * dig1 + dig2

# Parameters
n = 10
random.seed(42)

# Generate data
for _ in range(dummyrepetitions):
    dummy = random.uniform(0, 1)

Corr = np.array([[0] * n for _ in range(n)], dtype=float)
for i in range(n):
    for j in range(n):
        Corr[i][j] = (-1) ** abs(i - j) / (abs(i - j) + 1)

ssigma = np.zeros((n, 1), dtype=float)
mmu = np.zeros((n, 1), dtype=float)

ssigma[0] = 2
mmu[0] = 3

for i in range(n - 1):
    ssigma[i + 1] = ssigma[i] + 2 * random.uniform(0, 1)
    mmu[i + 1] = mmu[i] + 1

ddiag = np.zeros((n, n), dtype=float)
np.fill_diagonal(ddiag, ssigma.flatten())
C2 = np.matmul(np.matmul(ddiag, Corr), ddiag)

# Ensure C is positive semi-definite
C = 0.5 * (C2 + C2.T)
eigvals = np.linalg.eigvals(C)
if np.any(eigvals < 0):
    C -= np.min(eigvals) * np.eye(n)

# μ is the expected returns vector
mu = mmu.flatten()


## TASK 1

def solve_task_1(r_values):
    sigmas = []
    mus = []

    for r in r_values:
        # Quadratic programming setup
        P = C
        q = np.zeros(n)
        G = -np.eye(n)
        h = np.zeros(n)
        A = np.vstack([mu, np.ones(n)])
        b = np.array([r, 1])

        # Print debug information
        print(f"Solving for r = {r}")
        print("P:", P)
        print("q:", q)
        print("G:", G)
        print("h:", h)
        print("A:", A)
        print("b:", b)

        # Solve QP problem
        x = solve_qp(P, q, G, h, A, b, solver='quadprog')
        if x is None:
            print(f"No solution found for r = {r}")
            continue

        sigma = np.sqrt(np.dot(x.T, np.dot(C, x)))
        mu_val = np.dot(mu.T, x)

        sigmas.append(sigma)
        mus.append(mu_val)

    return sigmas, mus

# Define r values
r_values = np.arange(2.00, 9.25, 0.25)

# Solve task 1
sigmas, mus = solve_task_1(r_values)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(sigmas, mus, marker='o')
plt.xlabel('σ (Risk)')
plt.ylabel('μ (Return)')
plt.title('Efficient Frontier')
plt.grid(True)
plt.show()


## TASK 2

def solve_task_2(r_values):
    sigmas = []
    mus = []

    for r in r_values:
        # Quadratic programming setup
        P = C
        q = np.zeros(n)
        G = -np.eye(n)
        h = np.zeros(n)
        A = np.vstack([mu, np.ones(n)])
        b = np.array([r, 1])

        # Modify the equality constraint to an inequality constraint
        G = np.vstack([G, np.ones(n)])
        h = np.hstack([h, [1]])

        # Solve QP problem
        x = solve_qp(P, q, G, h, A[:-1], b[:-1], solver='quadprog')
        if x is None:
            print(f"No solution found for r = {r}")
            continue

        sigma = np.sqrt(np.dot(x.T, np.dot(C, x)))
        mu_val = np.dot(mu.T, x)

        sigmas.append(sigma)
        mus.append(mu_val)

    return sigmas, mus

# Solve task 2
sigmas_2, mus_2 = solve_task_2(r_values)

# Plot results for Task 2
plt.figure(figsize=(10, 6))
plt.plot(sigmas_2, mus_2, marker='o')
plt.xlabel('σ (Risk)')
plt.ylabel('μ (Return)')
plt.title('Efficient Frontier (Task 2)')
plt.grid(True)
plt.show()


## TASK 3

def solve_task_3(r_values):
    sigmas = []
    mus = []

    for r in r_values:
        # Quadratic programming setup
        P = C
        q = np.zeros(n)
        G = -np.eye(n)
        h = np.zeros(n)
        A = np.vstack([mu, np.ones(n)])
        b = np.array([r, 1])

        # Modify the equality constraint to an inequality constraint
        G = np.vstack([G, -mu])
        h = np.hstack([h, -r])

        # Solve QP problem
        x = solve_qp(P, q, G, h, A[-1:], b[-1:], solver='quadprog')
        if x is None:
            print(f"No solution found for r = {r}")
            continue

        sigma = np.sqrt(np.dot(x.T, np.dot(C, x)))
        mu_val = np.dot(mu.T, x)

        sigmas.append(sigma)
        mus.append(mu_val)

    return sigmas, mus

# Solve task 3
sigmas_3, mus_3 = solve_task_3(r_values)

# Plot results for Task 3
plt.figure(figsize=(10, 6))
plt.plot(sigmas_3, mus_3, marker='o')
plt.xlabel('σ (Risk)')
plt.ylabel('μ (Return)')
plt.title('Efficient Frontier (Task 3)')
plt.grid(True)
plt.show()


## TASK 4

def solve_task_4(r_values):
    sigmas = []
    mus = []

    for r in r_values:
        # Quadratic programming setup
        P = C
        q = np.zeros(n)
        A = np.vstack([mu, np.ones(n)])
        b = np.array([r, 1])

        # Remove the non-negativity constraint
        G = None
        h = None

        # Solve QP problem
        x = solve_qp(P, q, G, h, A, b, solver='quadprog')
        if x is None:
            print(f"No solution found for r = {r}")
            continue

        sigma = np.sqrt(np.dot(x.T, np.dot(C, x)))
        mu_val = np.dot(mu.T, x)

        sigmas.append(sigma)
        mus.append(mu_val)

    return sigmas, mus

# Solve task 4
sigmas_4, mus_4 = solve_task_4(r_values)

# Plot results for Task 4
plt.figure(figsize=(10, 6))
plt.plot(sigmas_4, mus_4, marker='o')
plt.xlabel('σ (Risk)')
plt.ylabel('μ (Return)')
plt.title('Efficient Frontier (Task 4)')
plt.grid(True)
plt.show()