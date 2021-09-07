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
    sv_no_zeros = np.delete(sv, np.where(sv < 10**(-300)))
    print('Sv no zeros: ', sv_no_zeros)
    sv_min = sv_no_zeros[0]
    sv_len = np.shape(sv_no_zeros)[0]

    for i in range(sv_len):
        if sv_no_zeros[i] < sv_min:
            sv_min = sv_no_zeros[i]

    print('Sv min: ', sv_min)
    sv_max = max(sv)

    btm_bnd = floor(log10(sv_min))

    top_bnd = floor(log10(sv_max))
    print('btm: ', btm_bnd)
    print('top: ', top_bnd)

    # range(..., ...) can be changed
    count = [0 for i in range(abs(btm_bnd - top_bnd)+1)]

    for s in sv:
        if s>0:
            print('value: ', floor(log10(s)))
            count[floor(log10(s)) + abs(btm_bnd)] += 1

    #for i in range(21):
        #count[i] /= mat_size

    axx.plot([i for i in range(btm_bnd, top_bnd + 1)], count, '--')
    axx.set_xlabel('log_10(s)')
    axx.set_ylabel(f'$\sigma = {mat_var}$')
    axx.set_title(f'Depth {dep}')

if __name__ == '__main__':

    fig, axs = plt.subplots(2,2)

    fig.tight_layout(pad = 3)

    NeuralNetwork(10,axs[0,0], 5, 0.05)
    NeuralNetwork(20, axs[0,1], 5, 0.05)
    NeuralNetwork(30, axs[1,0], 5, 0.05)
    NeuralNetwork(50, axs[1,1], 5, 0.05)

    plt.show()
