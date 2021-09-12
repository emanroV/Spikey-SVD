import numpy as np
from numpy import loadtxt
from math import log10, floor, sqrt
from scipy.linalg import svdvals
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
from multiprocessing import Pool


def relu(x):
    return 0 if x<0 else x

def NeuralNetwork(mat_size,dep, axx, mat_var, bias_var):

    print(f'Mat_size = {mat_size}')

    # size of matrices
    # mat_size = 10

    # multiprocessing.cpu_count() = 8
    with Pool(8) as p:
        Weight_array = p.map(ortho_group.rvs, [mat_size for _ in range(dep)])
    #Weight_array = list(map(ortho_group.rvs, [mat_size for _ in range(dep)]))

    # print(Weight_array[0])

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

    sv = svd(Jacobi)[0]

    # print('Sv', sv)

    sv_no_zeros = np.delete(sv, np.where(sv < 10**(-12)))
    print(f'Len of sv_cut: {len(sv_no_zeros)}')
    # print('Sv no zeros: ', sv_no_zeros)
    # sv_min = sv_no_zeros[0]
    # sv_len = np.shape(sv_no_zeros)[0]

    # for i in range(sv_len):
    #    if sv_no_zeros[i] < sv_min:
    #        sv_min = sv_no_zeros[i]

    # print('Sv min: ', sv_min)
    sv_min = min(sv_no_zeros)
    sv_max = max(sv_no_zeros)
    print(f'SV_min: {sv_min}, SV_max: {sv_max}')

    # btm_bnd = floor(log10(sv_min))
    # top_bnd = floor(log10(sv_max))
    # print('btm: ', btm_bnd)
    # print('top: ', top_bnd)

    # range(..., ...) can be changed
    # count = [0 for i in range(abs(btm_bnd - top_bnd)+1)]

    # for s in sv:
    #    if s>0:
    #        print('value: ', floor(log10(s)))
    #        count[floor(log10(s)) - btm_bnd] += 1

    #for i in range(21):
        #count[i] /= mat_size

    # sv_hist = [floor(log10(sv_no_zeros[i])) for i in range(sv_len)]

    axx.hist (sv_no_zeros, floor(sqrt(len(sv_no_zeros))), (sv_min,sv_max), histtype='bar', rwidth = 1.0)
    # axx.hist(sv_hist, abs(btm_bnd - top_bnd) + 1, (btm_bnd, top_bnd), histtype = 'bar', rwidth = 0.9)
    # axx.plot([i for i in range(btm_bnd, top_bnd + 1)], count, '--')
    axx.set_ylabel('log_10(s)')
    # axx.set_xlabel(f'Max: {floor(log10(sv_max))}, Min: {floor(log10(sv_min))}')
    axx.set_xlabel(f'Pos: {np.shape(sv_no_zeros)[0]}, Max: {round(sv_max,3)}, Min: {round(sv_min,3)}')
    axx.set_title(f'N = {mat_size}')

if __name__ == '__main__':

    fig, axs = plt.subplots(3,3)

    fig.suptitle('On Curve', fontsize = 14)

    fig.tight_layout(pad = 3)

    data = loadtxt('relu_critical.csv', delimiter=',')

    data_len = np.shape(data)[0]

    sw_sb = np.random.randint(0,data_len-1)
    # NeuralNetwork(1,axs[0,0], data[sw_sb][0], data[sw_sb][1])
    # NeuralNetwork(2,axs[0,1], data[sw_sb][0], data[sw_sb][1])
    # NeuralNetwork(3,axs[0,2], data[sw_sb][0], data[sw_sb][1])
    # NeuralNetwork(4,axs[1,0], data[sw_sb][0], data[sw_sb][1])
    # NeuralNetwork(5,axs[1,1], data[sw_sb][0], data[sw_sb][1])
    # NeuralNetwork(6,axs[1,2], data[sw_sb][0], data[sw_sb][1])
    # NeuralNetwork(7,axs[2,0], data[sw_sb][0], data[sw_sb][1])
    # NeuralNetwork(8,axs[2,1], data[sw_sb][0], data[sw_sb][1])
    # NeuralNetwork(9,axs[2,2], data[sw_sb][0], data[sw_sb][1])
    for i in range(9):
        NeuralNetwork(100*(i+1),10,axs[floor(i/3),i%3],data[sw_sb][0],data[sw_sb][1])

    plt.show()
