import numpy as np
from numpy import loadtxt
from math import log10, floor
from scipy.linalg import svdvals
from tensorflow.linalg import svd
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
from multiprocessing import Pool


def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )

def der_sig(x):
    return sigmoid(x) * ( 1 - sigmoid(x) )

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

    vec = np.random.randn(mat_size)

    Jacobi = np.identity(mat_size)

    sv_lst = []

    for i in range(dep):
        bias_vec = np.random.randn(mat_size)*bias_var
        h = Weight_array[i].dot(vec) + bias_vec
        for j in range(mat_size):
            D[i][j,j] = der_sig(h[j])
            vec[j] = sigmoid(h[j])

        Jacobi = np.matmul(np.matmul(Jacobi, D[i]), Weight_array[i])

        sv = svd(Jacobi)[0]
        sv_lst.append(floor(log10(np.mean(sv))))

    return sv_lst
    

if __name__ == '__main__':

    data = loadtxt('sigmoid_critical.csv', delimiter=',')

    data_len = np.shape(data)[0]

    sw_sb = np.random.randint(0,data_len-1)

    num_data = 200
    mean_sv = NeuralNetwork(num_data, data[sw_sb][0], data[sw_sb][1]) 

    plt.suptitle('Mean SV', fontsize = 14)
    plt.xlabel('Depth of Network')
    plt.plot([i for i in range(1,num_data+1)], mean_sv)
    plt.show()

