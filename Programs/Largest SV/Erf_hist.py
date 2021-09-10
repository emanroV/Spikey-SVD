from math import log10, floor, pi
import numpy as np
from numpy import loadtxt
from scipy.linalg import svdvals
from scipy.stats import ortho_group
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.special import erf



def NeuralNetwork(dep, mat_var, bias_var):

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
                D[i][j,j] = 2/np.sqrt(pi)*np.exp(-h[j]**2)
                vec[j] = erf(h[j])

    Jacobi = np.identity(mat_size)

    for i in range(dep):
        Jacobi = np.matmul(np.matmul(Jacobi, D[i]), Weight_array[i])

    sv = svdvals(Jacobi)
    return floor(log10(max(sv)))
    

if __name__ == '__main__':


    data = loadtxt('erf_critical.csv', delimiter=',')

    data_len = np.shape(data)[0]

    sw_sb = np.random.randint(0,data_len-1)

    num_data = 50
    largest_sv = [NeuralNetwork(i, data[sw_sb][0], data[sw_sb][1]) for i in range(2, num_data)]

    plt.suptitle('Largest Singular Value', fontsize = 14)
    plt.xlabel('Depth of Network')
    plt.plot([i for i in range(2,num_data)], largest_sv)
    plt.show()
