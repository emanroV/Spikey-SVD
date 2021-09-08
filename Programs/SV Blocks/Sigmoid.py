import numpy as np
from numpy import loadtxt
from math import log10, floor
from scipy.linalg import svdvals
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
from multiprocessing import Pool


def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )

def der_sig(x):
    return sigmoid(x) * ( 1 - sigmoid(x) )

def NeuralNetwork(dep, mat_var, bias_var):

    # size of matrices
    mat_size = 200

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
            D[i][j,j] = der_sig(h[j])
            vec[j] = sigmoid(h[j])

    Jacobi = np.identity(mat_size)

    for i in range(dep):
        # J = D_1*W_1 * D_2*W_2 * ... 
        Jacobi = np.matmul(np.matmul(Jacobi, D[i]), Weight_array[i])

    sv = svdvals(Jacobi)
    sv_list = []
    for s in sv:
        if s > 0:
            sv_list.append(floor(log10(s)))

    unique_sv = set(sv_list)
    sv_len = len(unique_sv)
    SV_Blocks = np.zeros((sv_len, 2))

    i=0
    for s in unique_sv:
        SV_Blocks[i][0] = s
        SV_Blocks[i][1] = sv_list.count(s)
        i += 1

    print(SV_Blocks)
    print('---------------------')
    mean = 0
    for i in range(sv_len):
        mean += SV_Blocks[i][1]
    print('Mean size of Blocks: ', mean / sv_len)
    print('====================')

if __name__ == '__main__':

    data = loadtxt('linear_critical.csv', delimiter=',')

    data_len = np.shape(data)[0]

    sw_sb = np.random.randint(0,data_len-1)
    NeuralNetwork(10, data[sw_sb][0], data[sw_sb][1])
    NeuralNetwork(20, data[sw_sb][0], data[sw_sb][1])
    NeuralNetwork(30, data[sw_sb][0], data[sw_sb][1])
    NeuralNetwork(50, data[sw_sb][0], data[sw_sb][1])

