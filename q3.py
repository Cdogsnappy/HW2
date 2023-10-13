import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0, 1], [-26, - 2]])

steps = 150
h = .02


def fun(t):
    return (1 / 5) * np.exp(-t) * (np.sin(5 * t) + 5 * np.cos(5 * t))


def der(t):
    return (-26 / 5) * np.exp(-t) * np.sin(5 * t)


def approx():
    lhs = np.identity(2) - (h / 3) * A
    #y_1 and y_2
    y_k_1 = np.array([fun(0), der(0)])
    y_k_2 = np.array([fun(.02), der(.02)])
    y_sol = [y_k_1, y_k_2]
    step = 1#For indexing purposes
    while step < steps + 1:
        b_k = (h / 3) * (np.dot(A, y_sol[step-1]) - 4 * np.dot(A, y_sol[step]))
        rhs = y_sol[step-1] + b_k
        y_sol.append(np.linalg.solve(lhs, rhs))
        step += 1

    mesh = np.linspace(0,3,152)
    y_sol = np.array(y_sol)
    plt.plot(mesh,y_sol[:,1])
    plt.plot(mesh,fun(mesh),color='r')
    plt.show()

approx()
