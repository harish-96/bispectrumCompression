import matplotlib.pyplot as plt     
from BFisherutils import *
import pdb


def KL_matrix(mean_der, cov):
    return np.dot(mean_der.T, np.linalg.inv(cov))

def conv1d_matrix(shape, kernel):
    # shape[0] * len(kernel) == shape[1]
    A = np.zeros(shape)
    for i in range(shape[0]):
        A[i, i*len(kernel):(i+1)*len(kernel)] = kernel
    return A

def Fii_bin(K, mu, params, navg, Vs, par_index=0, bin_size=8):
    dmu = mu[1] - mu[0]
    Nmu = len(mu)
    Nk = len(K)
    faa_bin = 0
    eps = 1e-6

    Nk = int(Nk - Nk%bin_size)
    K = K[:Nk]
    if Nk == 1:
        dk = K[0]
    else:
        dk = K[1] - K[0]

    Nk_t = int(Nk//bin_size)
    kernel = np.ones(bin_size)/bin_size
    A = conv1d_matrix((Nk_t, Nk), kernel)

    K_t = np.dot(A, K)
    if Nk_t == 1:
        dk_t = K_t[0]
    else:
        dk_t = K_t[1] - K_t[0]

    Ps = np.zeros((Nk, Nmu), dtype=np.float64)
    dPda = np.zeros((Nk,Nmu), dtype=np.float64)
    Covs = np.zeros((Nk,Nmu), dtype=np.float64)

    CovP_t = np.zeros((Nk_t, Nk_t, Nmu))
    Fp_t = np.zeros((Nk_t, Nk_t, Nmu))
    dP_tda = np.zeros((Nk_t, Nmu))

    dparams = np.zeros_like(params)
    dparams[par_index] = eps

    for l in range(Nmu):
        for i in range(Nk):
            Ps[i, l] = Pk((K[i], mu[l]), params)
            Covs[i,l] = CovP((K[i], mu[l]), params, (navg, Vs))
            Covs[i, l] *= dk**2 / (2*np.pi*K[i]**2*dmu)
            dPda[i,l] = (Pk((K[i], mu[l]), params+dparams) - Ps[i,l]) / eps
        dP_tda[:, l] = np.dot(A, dPda[:,l])
        CovP_t[:, :, l] = np.dot(A, np.dot(np.diag(Covs[:,l]), A.T))
        Fp_t[:,:,l] = np.linalg.inv(CovP_t[:, :, l])
        
        faa_bin += np.sum(dP_tda[:, l]**2 * np.diagonal(Fp_t[:,:,l]))

    return faa_bin

def F_KL(K, mu, params, navg, Vs, par_indices=[0]):
    dmu = mu[1] - mu[0]
    dk = K[1] - K[0]
    Nmu = len(mu)
    Nk = len(K)
    eps = 1e-6
    Ps = np.zeros(Nk, dtype=np.float64)
    dPda = np.zeros((Nk,len(par_indices)), dtype=np.float64)
    Covs = np.zeros(Nk, dtype=np.float64)

    P_func = Pk_new
    C_func = CovP_new

    # for i in range(Nk):
    #     for l in range(Nmu):
    #         Ps[i] += P_func((K[i], mu[l]), params) / Nmu
    #         Covs[i] += C_func((K[i], mu[l]), params, (navg, Vs)) / Nmu**2
    Ps = np.mean(Pk_vec((K, mu), params), axis=1)
    Covs = np.mean(CovP_vec((K, mu), params, (navg, Vs)), axis=1)/Nmu
    Covs *= dk**2 / (2*np.pi*K**2*dmu)

    for j, par_index in enumerate(par_indices):
        dparams = np.zeros_like(params, dtype=np.float64)
        dparams[par_index] = eps
        dPda[:, j] = np.mean((Pk_vec((K, mu), params+dparams) - Pk_vec((K,mu), params))/eps, axis=1)
        # for i in range(Nk):
        #     for l in range(Nmu):
        #         dPda[i, j] += (P_func((K[i], mu[l]), params+dparams) - P_func((K[i], mu[l]), params)) / eps / Nmu

    faa = np.dot(dPda.T, np.dot(np.diag(1/Covs), dPda))
    B = KL_matrix(dPda, np.diag(Covs))
    f_t = np.linalg.inv(np.dot(B, np.dot(np.diag(Covs), B.T)))
    faa_kl = np.dot(dPda.T, np.dot(B.T, np.dot(f_t, np.dot(B, dPda))))
    pdb.set_trace()


    return faa_kl


def Fii(K, mu, params, navg, Vs, par_index=0):
    dmu = mu[1] - mu[0]
    dk = K[1] - K[0]
    Nmu = len(mu)
    Nk = len(K)
    faa = 0
    eps = 1e-6
    Ps = np.zeros((Nk, Nmu), dtype=np.float64)
    dPda = np.zeros((Nk,Nmu), dtype=np.float64)
    Covs = np.zeros((Nk,Nmu), dtype=np.float64)

    dparams = np.zeros_like(params)
    dparams[par_index] = eps

    for l in range(Nmu):
        for i in range(Nk):
            Ps[i, l] = Pk((K[i], mu[l]), params)
            Covs[i,l] = CovP((K[i], mu[l]), params, (navg, Vs))
            dPda[i,l] = (Pk((K[i], mu[l]), params+dparams) - Ps[i,l]) / eps
            faa += 2*np.pi* K[i]**2 * dmu * dPda[i,l]**2 / Covs[i,l] / dk**2

    return faa


if __name__ == "__main__":
    apar = 1.01
    aper = 0.99
    f = 0.4
    b1 = 1.7
    b2 = 1
    params = (apar, aper, f, b1, b2)

    navg = 0.01
    Vs = 1
    eps = 1e-6
    Kmax, Kmin = (0.2, 1e-3)
    Nk = 100
    bin_sizes = [1, 2, 4, 5, 10, 20, 25, 50, 100]
    mu = np.linspace(-1, 1, 200)

    dk = (Kmax - Kmin) / Nk
    K = np.arange(Kmin, Kmax, dk)

    Fii_t = np.zeros(len(bin_sizes))

    # for j in range(len(bin_sizes)):

    #     Fii_t[j] = Fii_bin(K, mu, params, navg, Vs, 0, bin_sizes[j])

    #     print(Fii_t[j]) 

    # plt.plot(bin_sizes, Fii_t, '*-')
    # plt.xlabel('Bin sizes')
    # plt.ylabel('$F_{aa}$')
    # plt.savefig('./plot_bins.png')
    # plt.show()

