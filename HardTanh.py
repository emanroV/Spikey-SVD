import numpy as np
from math import log10, floor
from scipy.linalg import svdvals
from scipy.stats import ortho_group
from multiprocessing import Pool
import matplotlib.pyplot as plt



def NeuralNetwork(dep, axx, mat_var, bias_var):

    # size of matrices
    mat_size = 100

    # multiprocessing.cpu_count() = 8
    with Pool(8) as p:
        Weight_array = p.map(ortho_group.rvs, [mat_size for _ in range(dep)])

    for i in range(dep):
        Weight_array[i] *= mat_var

    D = [np.identity(mat_size) for _ in range(dep)]

    vec = np.random.randn(mat_size)

    for i in range(dep):
        bias_vec = np.random.randn(mat_size) * bias_var
        h = Weight_array[i].dot(vec) + bias_vec
        for j in range(mat_size):
            if h[j]<-1:
                D[i][j,j] = 0
                vec[j]=-1
            elif h[j]>1:
                D[i][j,j] = 0
                vec[j]=1
            else:
                vec[j] = h[j]

    Jacobi = np.identity(mat_size)

    for i in range(dep):
        Jacobi = np.matmul(np.matmul(Jacobi, D[i]), Weight_array[i])

    sv = svdvals(Jacobi)
    print('check')

    print('----------------------------')

    print(sv)

    # range(..., ...) can be changed
    count = [0 for i in range(-100, 0)]
    for s in sv:
        if s>0:
            count[floor(log10(s))+100] += 1

    #for i in range(21):
        #count[i] /= mat_size

    # Plot setup
    axx.plot([i for i in range(-100, 0)], count, '--')
    axx.set_xlabel('log_10(s)')
    axx.set_ylabel(f'$\sigma^2 = {mat_var}$')
    axx.set_title(f'Depth {dep}')

if __name__ == '__main__':

    fig, axs = plt.subplots(2,2)

    fig.tight_layout(pad = 3)

    NeuralNetwork(10,axs[0,0], 0.1, 0.05)
    NeuralNetwork(20, axs[0,1], 0.1, 0.05)
    NeuralNetwork(30, axs[1,0], 0.1, 0.05)
    NeuralNetwork(50, axs[1,1], 0.1, 0.05)

    plt.show()
