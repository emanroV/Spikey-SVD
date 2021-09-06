import numpy as np
from math import log10, floor
from scipy.linalg import svdvals
import matplotlib.pyplot as plt
from multiprocessing import Pool


def NeuralNetwork(dep, axx, mat_var, bias_var):
    
    mat_size = 100

    Weight_array = [np.random.randn(mat_size,mat_size) for _ in range(dep)]

    for i in range(dep):
        Weight_array[i] *= mat_var


    Jacobi = np.identity(mat_size)

    vec = np.random.randn(mat_size)

    for i in range(dep):
        bias_vec = np.random.randn(mat_size) * bias_var
        h = Weight_array[i].dot(vec) + bias_vec
        for j in range(mat_size):
            vec[j] = h[j]
        Jacobi = np.matmul(Jacobi, Weight_array[i])

    sv = svdvals(Jacobi)

    print(sv)

    count = [0 for i in range(-60, 60)]

    for s in sv:
        if s>0:
            count[floor(log10(s))+60] += 1

    #for i in range(200):
        #count[i] /= mat_size

    axx.plot([i for i in range(-60, 60)], count, '--')
    axx.set_xlabel('log_10(s)')
    axx.set_ylabel(f'$\sigma^2 = {mat_var}$')
    axx.set_title(f'Depth {dep}')

if __name__ == "__main__":
    fig, axs = plt.subplots(2,2)

    fig.tight_layout(pad=3.0)

    np.random.RandomState(100)

    NeuralNetwork(10,axs[0,0], 0.1, 0.05)
    NeuralNetwork(20, axs[0,1], 0.1, 0.05)
    NeuralNetwork(30, axs[1,0], 0.1, 0.05)
    NeuralNetwork(50, axs[1,1], 0.1, 0.05)

    plt.show()
