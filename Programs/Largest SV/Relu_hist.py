import numpy as np
from numpy import loadtxt
from math import log10, floor
from scipy.linalg import svdvals
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
from multiprocessing import Pool


def relu(x):
    return 0 if x<0 else x

def NeuralNetwork(dep, mat_var, bias_var):

    # size of matrices
    mat_size = 100

    # multiprocessing.cpu_count() = 8
    with Pool(8) as p:
        #Weight_array = p.map(ortho_group.rvs, [1000 for _ in range(dep)])
        Weight_array = p.map(ortho_group.rvs, [mat_size for _ in range(dep)])

    for i in range(dep):
        Weight_array[i] *= mat_var

    D = [np.identity(mat_size) for _ in range(dep)]

    Jacobi = np.identity(mat_size)

    vec = np.random.randn(mat_size)

    sv_lst = []

    for i in range(dep):
        bias_vec = np.random.randn(mat_size)*bias_var
        h = Weight_array[i].dot(vec) + bias_vec
        for j in range(mat_size):
            if h[j]<0:
                D[i][j,j] = 0
                vec[j]=0
            else:
                # phi^\prime = 1 = identity - matrix - entry
                vec[j] = h[j]
        Jacobi = np.matmul(np.matmul(Jacobi, D[i]), Weight_array[i])

        sv = svdvals(Jacobi)
        sv_lst.append(floor(log10(max(sv))))

    return sv_lst
    
if __name__ == '__main__':

    data = loadtxt('relu_critical.csv', delimiter=',')

    data_len = np.shape(data)[0]

    sw_sb = np.random.randint(0,data_len-1)

    num_data = 200
    max_sv = NeuralNetwork(num_data, data[sw_sb][0], data[sw_sb][1]) 

    plt.suptitle('Mean SV', fontsize = 14)
    plt.xlabel('Depth of Network')
    plt.plot([i for i in range(1,num_data+1)], max_sv)
    plt.show()
