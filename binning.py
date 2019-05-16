import matplotlib.pyplot as plt     
from BFisherutils import *
import pdb


def compFaa(K, mu, Nk, Nmu, apar, aper, f, b1, b2, navg, Vs, bin_size=8):
    Nk = int(Nk - Nk%bin_size)
    K = K[:Nk]
    if Nk == 1:
        dk = K[0]
    else:
        dk = K[1] - K[0]
    dmu = mu[1] - mu[0]
    faa, faa_t = 0, 0
    eps = 1e-6
    Ps = np.zeros((Nk, Nmu), dtype=np.float64)
    dPda = np.zeros((Nk,Nmu), dtype=np.float64)
    Covs = np.zeros((Nk,Nmu), dtype=np.float64)

    Nk_t = int(Nk//bin_size)
    # pdb.set_trace()
    A = conv1d_matrix((Nk_t, Nk), np.ones(bin_size)/bin_size)
    K_t = np.dot(A, K)
    if Nk_t == 1:
        dk_t = K_t[0]
    else:
        dk_t = K_t[1] - K_t[0]
    CovP_t = np.zeros((Nk_t, Nk_t, Nmu))
    Fp_t = np.zeros((Nk_t, Nk_t, Nmu))
    dP_tda = np.zeros((Nk_t, Nmu))

    for l in range(Nmu):
        for i in range(Nk):
            Ps[i, l] = Pk((K[i], mu[l]), (apar, aper, f, b1, b2))
            Covs[i,l] = CovP((K[i], mu[l]), (apar, aper, f, b1, b2), (navg, Vs))
            dPda[i,l] = (Pk((K[i], mu[l]), (apar+eps, aper, f, b1, b2)) - Ps[i,l]) / eps

    for l in range(Nmu):
        dP_tda[:, l] = np.dot(A, dPda[:,l])
        CovP_t[:, :, l] = np.dot(A, np.dot(np.diag(Covs[:,l]), A.T))
        Fp_t[:,:,l] = np.linalg.inv(CovP_t[:, :, l])


    for i in range(Nk_t):
        for k in range(Nmu):
            faa_t += 2*np.pi * dmu * K_t[i]**2 * dP_tda[i, k]**2 * Fp_t[i, i, k] / dk_t**2
    # pdb.set_trace()

    return faa_t

def conv1d_matrix(shape, kernel):
    # shape[0] * len(kernel) == shape[1]
    A = np.zeros(shape)
    for i in range(shape[0]):
        A[i, i*len(kernel):(i+1)*len(kernel)] = kernel
    return A

mu = np.linspace(-1, 1, 5)
apar = 1.01
aper = 0.99
f = 0.4
b1 = 1.7
b2 = 0
navg = 0.01
Vs = 1
eps = 1e-6
Kmax, Kmin = (0.8, 1e-3)
Nk = 100
parc = (apar, aper, f, b1, b2)

dk = (Kmax - Kmin) / Nk
K = np.arange(Kmin, Kmax, dk)

bin_sizes = [1, 2, 4, 5, 10, 20, 25, 50, 100]

Faa_t = np.zeros(len(bin_sizes))

for j in range(len(bin_sizes)):

    Faa_t[j] = compFaa(K, mu, Nk, len(mu), apar, aper, f, b1, b2, navg, Vs, bin_sizes[j])

    print(Faa_t[j]) 

plt.plot(bin_sizes, Faa_t, '*-')
plt.xlabel('Bin sizes')
plt.ylabel('$F_{aa}$')
plt.legend()
plt.savefig('./plot_bins.png')
plt.show()

