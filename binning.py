import matplotlib.pyplot as plt     
from numba import jit
from BFisherutils import *
from scipy.ndimage import convolve1d
import pdb

@jit
def compFaa(K, mu, Nk, Nmu, apar, aper, f, b1, b2, navg, Vs, bin_size=3):
    dk = K[1] - K[0]
    dmu = mu[1] - mu[0]
    eps = 1e-6
    Ps = np.zeros((Nk, Nmu), dtype=np.float64)
    dPda = np.zeros((Nk,Nmu), dtype=np.float64)
    Covs = np.zeros((Nk,Nmu), dtype=np.float64)
    notTriangles = 0
    faa, faa_t = 0, 0
    for i in range(Nk):
        for l in range(Nmu):
            Ps[i, l] = Pk((K[i], mu[l]), (apar, aper, f, b1, b2))
            Covs[i,l] = CovP((K[i], mu[l]), (apar, aper, f, b1, b2), (navg, Vs))
            dPda[i,l] = (Pk((K[i], mu[l]), (apar+eps, aper, f, b1, b2)) - Ps[i,l]) / eps
            dFaa = K[i]**2 * dk * dmu * dPda[i,l]**2 / Covs[i,l]
            if not np.isnan(dFaa):
                faa += dFaa
            else:
                notTriangles += 1

    P_t, A = conv1d(Ps, np.ones(bin_size)/bin_size)
    CovP_t = np.zeros((P_t.shape[0], P_t.shape[0], P_t.shape[1]))
    Fp_t = np.zeros((P_t.shape[0], P_t.shape[0], P_t.shape[1]))
    dP_tda = np.zeros((Nk//bin_size, Nmu))
    for i in range(Nmu):
        CovP_t[:, :, i] = np.dot(A, np.dot(np.diag(Covs[:,i]), A.T))
        Fp_t[:,:,i] = np.linalg.inv(CovP_t[:, :, i])
        dP_tda[:, i] = np.dot(A, dPda[:,i])
    
    for i in range(Nk):
        for j in range(Nk):
            for k in range(Nmu):
                faa_t += dP_tda[i, k] * dP_tda[j, k] * Fp_t[i, j, k]
    
    return faa, faa_t

def conv1d(M, kernel):
    A = np.zeros((int(M.shape[0]/len(kernel)), M.shape[0]))
    for i in range(A.shape[0]):
        A[i, i*len(kernel):(i+1)*len(kernel)] = kernel
    conv = np.zeros((int(M.shape[0]/len(kernel)), M.shape[1]))
    for i in range(M.shape[1]):
        conv[:, i] = np.dot(A, M[:,i])
    return conv, A

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

Kmax, Kmin = (0.3, 1e-3)
Faa, dk = [], []
for Nk in range(10, 30, 5):
    dk.append((Kmax - Kmin) / Nk)
    Faa.append(0)

    K = np.arange(Kmin, Kmax, dk[-1])

    Faa[-1], _ = compFaa(K, mu, Nk, len(mu), apar, aper, f, b1, b2, navg, Vs)

    print(Faa[-1], _) 

plt.plot(dk, Faa, '*-')
plt.xlabel('$\Delta k$')
plt.ylabel('$F_{aa}$')
plt.show()

# plt.plot(Ks, B1s, label='Bispectrum'); plt.xlabel('k1');plt.ylabel('Bisp(k1, 0.25, 0.25,pi/15,pi/15)'), plt.legend()
# plt.show()
# plt.plot(Ks, Covs, label='Covariance'); plt.xlabel('k1');plt.ylabel('CovB(k1, 0.25, 0.25,pi/15,pi/15)'), plt.legend()
# plt.show()
# plt.plot(Ks, B1s/Covs, label='SNR'); plt.xlabel('k1');plt.ylabel('snr(k1, 0.25, 0.25,pi/15,pi/15)');plt.legend()
# plt.show()


#k1k2k3*dK**3*dmu*dphi
