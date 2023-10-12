import numpy as np

s = np.sqrt(3) / 2


def ind(i, j, n):
    return i - 1 + (j - 1) * (2 * n - 2 - j) // 2


def mul_A(u, n):  # calculates A*u without building A
    h = 1 / (n + 1)
    m = len(u)
    hfac = 2 / (h ** 2)
    Au = np.zeros((m))
    for j in range(1, n):
        for i in range(1, n - j):
            ij = ind(i, j, n)
            Au[ij] += -6 * u[ij]
            if i > 0:
                Au[ij] += u[ind(i - 1, j, n)]
                if j < n - 1:
                    Au[ij] += u[ind(i - 1, j + 1, n)]
            if i < n - j - 1:
                Au[ij] += u[ind(i + 1, j, n)]
                if j > 0:
                    Au[ij] += u[ind(i + 1, j - 1, n)]
            if j > 0:
                Au[ij] += u[ind(i, j - 1, n)]
            if j < n - 3:
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
        u += np.dot(a, p)
        r -= np.dot(a, w)
        rr_next = np.dot(r.T, r)
        if np.linalg.norm(r) < 1e-10:
            break
        print(np.linalg.norm(r))
        p = r + (rr_next / rr_old) * p
        rr_old = rr_next
        k += 1
    return k


def inrange(i, j, n):
    return i > 0 and j > 0 and i + j < n


def precond(d, n):
    m = (n - 1) * (n - 2) // 2
    A = np.zeros((m, m))
    b_s = np.floor(np.sqrt(n))
    for i in range(np.floor(n / b_s)):  # Extracting diagonal blocks from the dense matrix and inverting them block-wise
        A[0 + i * b_s:b_s + i * b_s, 0 + i * b_s:b_s + i * b_s] = np.linalg.inv(
            d[0 + i * b_s:b_s + i * b_s, 0 + i * b_s:b_s + i * b_s])
    A[np.floor(n / b_s) * b_s:m - 1, np.floor(n / b_s) * b_s:m - 1] = np.linalg.inv(
        d[np.floor(n / b_s) * b_s:m - 1, np.floor(n / b_s) * b_s:m - 1])
    return A


def pre_tri_cg(n):
    h = 1. / n
    hfac = 1 / (3 * h ** 2)
    m = (n - 1) * (n - 2) // 2
    d = np.zeros((m, m))
    r = np.zeros((m))
    for j in range(1, n):
        for i in range(1, n - j):
            ij = ind(i, j, n)
            # Derivative matrix
            d[ij, ij] = -12 * hfac
            if inrange(i + 1, j, n): d[ij, ind(i + 1, j, n)] = 2 * hfac
            if inrange(i, j - 1, n): d[ij, ind(i, j - 1, n)] = 2 * hfac
            if inrange(i - 1, j + 1, n): d[ij, ind(i - 1, j + 1, n)] = 2 * hfac
            r[ij] = fun(h * (i + 0.5 * j), h * s * j)
    A = precond(d, n)
    u = np.zeros(m)
    p = np.dot(A, r)
    y = np.copy(p)
    rr_old = np.dot(y.T, r)
    k = 0
    while True:
        z = mul_A(p, n)
        a = rr_old / np.dot(z, p)
        u += a * p
        r -= a * z
        rr_next = np.dot(r.T, r)
        if res(r) < 1e-10:
            break
        p = r + (rr_next / rr_old) * p
        rr_old = rr_next
        k += 1
    return k


def res(r):  # Our 2-norm def
    return np.sqrt((s / (2 * n ** 2)) * np.sum(np.square(r)))


def fun(x, y):
    return -8 * (np.sqrt(3) - 2 * y) * np.cos(y) + 4 * (-4 - 3 * x + 3 * x * x + np.sqrt(3) * y - y * y) * np.sin(y)


for n in (10, 20, 40, 80, 160):
    print(tri_cg(n))
    print("preconditioned: " + str(pre_tri_cg(n)))
