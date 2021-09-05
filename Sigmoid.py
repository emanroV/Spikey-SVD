import numpy as np
from math import log10, floor
from scipy.linalg import svdvals
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
from multiprocessing import Pool



def NeuralNetwork(dep, axx, mat_var, bias_var):
    def sigmoid(x):
        return 1 / ( 1 + np.exp(-x) )

    def der_sig(x):
        return sigmoid(x) * ( 1 - sigmoid(x) )

    # size of matrices
    mat_size = 1000

    # multiprocessing.cpu_count() = 8
    with Pool(8) as p:
        #Weight_array = p.map(ortho_group.rvs, [1000 for _ in range(dep)])
        Weight_array = p.map(ortho_group.rvs, [mat_size for _ in range(dep)])

    for i in range(dep):
        Weight_array[i] *= mat_var

    D = [np.identity(mat_size) for _ in range(dep)]

    vec = np.random.randn(mat_size)

    for i in range(dep):
        bias_vec = np.random.randn(mat_size)*bias_var
        h = Weight_array[i].dot(vec) + bias_vec
        for j in range(mat_size):
            D[i][j,j] = der_sig(h[i])
            vec[j] = sigmoid(h[i])

    Jacobi = np.identity(mat_size)

    for i in range(dep):
        # J = D_1*W_1 * D_2*W_2 * ... 
        Jacobi = np.matmul(np.matmul(Jacobi, D[i]), Weight_array[i])

    sv = svdvals(Jacobi)
    print('check')

    print('---------------------------------------------------')
    print(sv)

    # range(..., ...) can be changed
    count = [0 for i in range(-200, 10)]
    for s in sv:
        # log_10 not defined for case s == 0
        if s>0:
            count[floor(log10(s))+200] += 1


    # Plot setup
    axx.plot([i for i in range(-200, 10)], count, '--')
    axx.set_xlabel('log_10(s)')
    axx.set_ylabel(f'$\sigma^2 = {mat_var}$')
    axx.set_title(f'Depth {dep}')

if __name__ == '__main__':

    fig, axs = plt.subplots(2,2)

    fig.tight_layout(pad = 3)

    np.random.RandomState(100)

    #NeuralNetwork(5,axs[0,0], 0.01, 0.05)
    #NeuralNetwork(50, axs[0,1], 0.01, 0.05)
    #NeuralNetwork(100, axs[1,0], 0.01, 0.05)
    #NeuralNetwork(150, axs[1,1], 0.01, 0.05)
    NeuralNetwork(10,axs[0,0], 0.1, 0.05)
    NeuralNetwork(20, axs[0,1], 0.5, 0.05)
    NeuralNetwork(50, axs[1,0], 2, 0.05)
    NeuralNetwork(50, axs[1,1], 5, 0.05)
    #NeuralNetwork(5,axs[0,0], 2, 0.05)
    #NeuralNetwork(5, axs[0,1], 50, 0.05)
    #NeuralNetwork(5, axs[1,0], 100, 0.05)
    #NeuralNetwork(5, axs[1,1], 200, 0.05)

    plt.show()
