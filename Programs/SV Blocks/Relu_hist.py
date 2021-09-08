import numpy as np
from numpy import loadtxt
from math import log10, floor
from scipy.linalg import svdvals
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
from multiprocessing import Pool


def relu(x):
    return 0 if x<0 else x

def NeuralNetwork(dep, axx, mat_var, bias_var):

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
            if h[j]<0:
                D[i][j,j] = 0
                vec[j]=0
            else:
                # phi^\prime = 1 = identity - matrix - entry
                vec[j] = h[j]

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

    mean = format(mean / sv_len, '.2f')
    print('Mean size of Blocks: ', mean)
    print('====================')

    sv_min = int(min(SV_Blocks[:,0]))
    sv_max = int(max(SV_Blocks[:,0]))
    print('SV min: ', sv_min)
    print('SV max: ', sv_max)

    axx.hist(sv_list, abs(sv_max - sv_min)+1, (sv_min, sv_max), histtype = 'bar', rwidth = 0.9)
    axx.set_xlabel('log_10(s)')
    axx.set_ylabel(f'Mean: {mean}')
    axx.set_title(f'Depth {dep}')


if __name__ == '__main__':

    fig, axs = plt.subplots(2,2)

    fig.suptitle('Occurences of singular values (floor(log10(s)))', fontsize = 14)

    fig.tight_layout(pad = 3.0)

    np.random.RandomState(100)

    data = loadtxt('relu_critical.csv', delimiter=',')

    data_len = np.shape(data)[0]

    sw_sb = np.random.randint(0,data_len-1)
    NeuralNetwork(10, axs[0,0], data[sw_sb][0], data[sw_sb][1])
    NeuralNetwork(20, axs[0,1], data[sw_sb][0], data[sw_sb][1])
    NeuralNetwork(30, axs[1,0], data[sw_sb][0], data[sw_sb][1])
    NeuralNetwork(50, axs[1,1], data[sw_sb][0], data[sw_sb][1])
    
    plt.show()
