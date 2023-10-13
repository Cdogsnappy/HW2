import time

import numpy as np
import math
import matplotlib.pyplot as plt

import time_test

s = np.sqrt(3) / 2


def ind(i, j, n):
    return i - 1 + (j - 1) * (2 * n - 2 - j) // 2


def mul_A(u, n):  # calculates A*u without building A
    h = 1 / (n + 1)
    hfac = 2 / (h ** 2)
    Au = -6 * u
    for j in range(1, n):
        for i in range(1, n - j):
            ij = ind(i, j, n)
            if inrange(i - 1, j, n):
                Au[ij] += u[ind(i - 1, j, n)]
            if inrange(i - 1, j + 1, n):
                Au[ij] += u[ind(i - 1, j + 1, n)]
            if inrange(i + 1, j, n):
                Au[ij] += u[ind(i + 1, j, n)]
            if inrange(i + 1, j - 1, n):
                Au[ij] += u[ind(i + 1, j - 1, n)]
            if inrange(i, j - 1, n):
                Au[ij] += u[ind(i, j - 1, n)]
            if inrange(i, j + 1, n):
                Au[ij] += u[ind(i, j + 1, n)]
    Au *= hfac
    return Au


def tri_cg(n):
    m = (n - 1) * (n - 2) // 2  # Interior points
    h = 1. / n
    u = np.zeros((m))
    r = np.zeros((m))
    for j in range(1, n):
        for i in range(1, n - j):
            r[ind(i, j, n)] = fun(h * (i + 0.5 * j), h * s * j)
    r -= mul_A(u, n)
    p = np.copy(r)
    rr_old = np.dot(r.T, r)
    k = 0
    while True:
        w = mul_A(p, n)
        a = rr_old / np.dot(p.T, w)
        u += a * p
        r -= a * w
        rr_next = np.dot(r.T, r)
        if res(r, n) < 1e-10:
            break
        p = r + (rr_next / rr_old) * p
        rr_old = rr_next
        k += 1
    return k


def inrange(i, j, n):
    return i > 0 and j > 0 and i + j < n


def precond(d, n):
    m = (n - 1) * (n - 2) // 2
    A = np.zeros((m, m))
    b_s = math.floor(np.sqrt(n))
    num_blocks = math.ceil(n / b_s)
    for i in range(num_blocks):  # Extracting diagonal blocks from the dense matrix and inverting them block-wise
        A[0 + i * b_s:b_s + i * b_s, 0 + i * b_s:b_s + i * b_s] = np.linalg.inv(d[0 + i * b_s:b_s + i * b_s, 0 + i * b_s:b_s + i * b_s])
    A[num_blocks * b_s:m, num_blocks * b_s:m] = np.linalg.inv(
        d[num_blocks * b_s:m, num_blocks * b_s:m])
    return A


def pre_tri_cg(n):
    h = 1. / n
    hfac = 1 / (h ** 2)
    m = (n - 1) * (n - 2) // 2
    d = np.zeros((m, m))
    r = np.zeros(m)
    for j in range(1, n):
        for i in range(1, n - j):
            ij = ind(i, j, n)
            d[ij, ij] = -12 * hfac
            if inrange(i + 1, j, n): d[ij, ind(i + 1, j, n)] = 2 * hfac
            if inrange(i, j - 1, n): d[ij, ind(i, j - 1, n)] = 2 * hfac
            if inrange(i - 1, j + 1, n): d[ij, ind(i - 1, j + 1, n)] = 2 * hfac
            if inrange(i - 1, j, n): d[ij, ind(i - 1, j, n)] = 2 * hfac
            if inrange(i, j + 1, n): d[ij, ind(i, j + 1, n)] = 2 * hfac
            if inrange(i + 1, j - 1, n): d[ij, ind(i + 1, j - 1, n)] = 2 * hfac
            r[ij] = fun(h * (i + 0.5 * j), h * s * j)
    A = precond(d, n)
    u = np.zeros(m)
    p = np.dot(A, r)
    y = np.copy(p)
    rr_old = np.dot(y.T, r)
    k = 0
    while True:
        z = mul_A(p, n)
        a = rr_old / np.dot(p.T, z)
        u += a * p
        r -= a * z
        if res(r, n) < 1e-10:
            break
        y = np.dot(A, r)
        rr_next = np.dot(y.T, r)
        p = y + (rr_next / rr_old) * p
        rr_old = rr_next
        k += 1
    return k


def res(r, n):  # Our 2-norm def
    return np.sqrt((s / (2 * n ** 2)) * np.sum(np.square(r)))


def fun(x, y):
    return -8 * (np.sqrt(3) - 2 * y) * np.cos(y) + 4 * (-4 - 3 * x + 3 * x * x + np.sqrt(3) * y - y * y) * np.sin(y)


def uex(x, y):
    return ((2 * y - np.sqrt(3)) ** 2 - 3 * (2 * x - 1) ** 2) * np.sin(y)



for n in (10,20,40,80,160):
    print("n = " + str(n) + ": " + str(tri_cg(n)) + " iterations")
    #print("preconditioned: " + str(n) + ": " + str(pre_tri_cg(n)))
def timer():
    times = []
    val_set = [10, 20, 40, 80, 160]
    for n in val_set:
        start = time.time()
        tri_cg(n)
        end = time.time()
        print("Time for n = " + str(n) + ": " + str(end - start) + " seconds")
        times.append(end - start)
    params = np.polyfit(np.log(val_set), np.log(times), 1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("T")
    print(params)
    plt.scatter(val_set, times)
    plt.plot(val_set, val_set ** (params[0]) * 2e-6, color='r')
    plt.show()



