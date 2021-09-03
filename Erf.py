from math import log10, floor, pi
import numpy as np
from scipy.linalg import svdvals
from scipy.stats import ortho_group
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.special import erf



def NeuralNetwork(dep, axx, var):
    def relu(x):
        return 0 if x<0 else x

    # size of matrices
    mat_size = 100

    # multiprocessing.cpu_count() = 8
    with Pool(8) as p:
        Weight_array = p.map(ortho_group.rvs, [mat_size for _ in range(dep)])

    for i in range(dep):
        Weight_array[i] *= var

    D = [np.identity(mat_size) for _ in range(dep)]

    vec = np.random.randn(mat_size)

    for i in range(dep):
        bias_vec = np.random.randn(mat_size)*0.05
        h = Weight_array[i].dot(vec) + bias_vec
        for j in range(mat_size):
                D[i][j,j] = 2/pi*np.exp(-h[j]**2)
                vec[j] = erf(h[j])

    Jacobi = np.identity(mat_size)

    for i in range(dep):
        Jacobi = np.matmul(np.matmul(Jacobi, D[i]), Weight_array[i])

    sv = svdvals(Jacobi)

    print('check')
    print('---------------------------------------')
    print(sv)

    # range(..., ...) can be changed
    count = [0 for i in range(-200, 100)]

    for s in sv:
        if s>0:
            count[floor(log10(s)) + 200] += 1

    #for i in range(21):
        #count[i] /= mat_size

    axx.plot([i for i in range(-200, 100)], count, '--')
    axx.set_xlabel('log_10(s)')
    axx.set_ylabel(f'$\sigma^2 = {var}$')
    axx.set_title(f'Depth {dep}')

if __name__ == '__main__':

    fig, axs = plt.subplots(2,2)

    fig.tight_layout(pad = 3)

    #NeuralNetwork(10,axs[0,0], 5)
    #NeuralNetwork(20, axs[0,1], 5)
    #NeuralNetwork(50, axs[1,0], 5)
    #NeuralNetwork(100, axs[1,1], 5)
    NeuralNetwork(50,axs[0,0], 0.5)
    NeuralNetwork(50, axs[0,1], 1)
    NeuralNetwork(50, axs[1,0], 3)
    NeuralNetwork(50, axs[1,1], 6)

    plt.show()
