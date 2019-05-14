import matplotlib.pyplot as plt     
from numba import jit
from BFisherutils import *
from scipy.ndimage import convolve1d
import pdb

def compFaa(K, mu, Nk, Nmu, apar, aper, f, b1, b2, navg, Vs, bin_size=5):
    dk = K[1] - K[0]
    dmu = mu[1] - mu[0]
    eps = 1e-6
    Ps = np.zeros((Nk, Nmu), dtype=np.float64)
    dPda = np.zeros((Nk,Nmu), dtype=np.float64)
    Covs = np.zeros((Nk,Nmu), dtype=np.float64)
    faa, faa_t = 0, 0

    Nk_t = Nk//bin_size
    A = conv1d_matrix((Nk_t, Nk), np.ones(bin_size)/bin_size)
    K_t = np.dot(A, K)
    CovP_t = np.zeros((Nk_t, Nk_t, Nmu))
    Fp_t = np.zeros((Nk_t, Nk_t, Nmu))
    dP_tda = np.zeros((Nk//bin_size, Nmu))

    for i in range(Nk):
        for l in range(Nmu):
            Ps[i, l] = Pk((K[i], mu[l]), (apar, aper, f, b1, b2))
            Covs[i,l] = CovP((K[i], mu[l]), (apar, aper, f, b1, b2), (navg, Vs))
            dPda[i,l] = (Pk((K[i], mu[l]), (apar+eps, aper, f, b1, b2)) - Ps[i,l]) / eps
            faa += K[i]**2 * dk * dmu * dPda[i,l]**2 / Covs[i,l]

    for i in range(Nmu):
        CovP_t[:, :, i] = np.dot(A, np.dot(np.diag(Covs[:,i]), A.T))
        Fp_t[:,:,i] = np.linalg.inv(CovP_t[:, :, i])
        dP_tda[:, i] = np.dot(A, dPda[:,i])
    
    for i in range(Nk//bin_size):
        for j in range(Nk//bin_size):
            for k in range(Nmu):
                faa_t += K_t[i]*dmu*K_t[j]*(K_t[1]-K_t[0])*dP_tda[i, k] * dP_tda[j, k] * Fp_t[i, j, k]
    
    return faa, faa_t

def conv1d_matrix(shape, kernel):
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

parc = (apar, aper, f, b1, b2)

Kmax, Kmin = (0.8, 1e-3)
Faa, dk, Faa_t = [], [], []
for Nk in range(10, 30, 2):
    dk.append((Kmax - Kmin) / Nk)
    Faa.append(0)
    Faa_t.append(0)

    K = np.arange(Kmin, Kmax, dk[-1])

    Faa[-1], Faa_t[-1] = compFaa(K, mu, Nk, len(mu), apar, aper, f, b1, b2, navg, Vs)

    print(Faa[-1], Faa_t[-1]) 

plt.plot(dk, Faa, '*-', label='$F_{aa}$')
plt.plot(dk, Faa_t, '*-', label=r'$\tildeF_{aa}$')
plt.xlabel('$\Delta k$')
plt.ylabel('$F_{aa}$')
plt.legend()
plt.show()

