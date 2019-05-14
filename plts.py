import matplotlib.pyplot as plt     
from numba import jit
from BFisherutils import *

@jit
def compFaa(K, mu, phi, Nk, Nmu, Nphi, apar, aper, f, b1, b2, navg, Vs):
    dk = K[1] - K[0]
    dmu = mu[1] - mu[0]
    dphi = phi[1] - phi[0]
    eps = 1e-6
    Bs = np.zeros((Nk, Nk, Nk, Nmu, Nphi), dtype=np.float64)
    dBda = np.zeros((Nk, Nk, Nk, Nmu, Nphi), dtype=np.float64)
    Covs = np.zeros((Nk, Nk, Nk, Nmu, Nphi), dtype=np.float64)
    notTriangles = 0
    faa = 0
    for i in range(Nk):
        for j in range(Nk):
            for k in range(Nk):
                for l in range(Nmu):
                    for m in range(Nphi):
                        Bs[i, j, k, l, m] = Bisp((K[i], K[j], K[k], mu[l], phi[m]), (apar, aper, f, b1, b2), (navg, Vs))
                        Covs[i, j, k, l, m] = CovB((K[i], K[j], K[k], mu[l], phi[m]), (apar, aper, f, b1, b2), (navg, Vs))
                        dBda[i, j, k, l, m] = (Bisp((K[i], K[j], K[k], mu[l], phi[m]), (apar+eps, aper, f, b1, b2), (navg, Vs)) - Bs[i, j, k, l, m]) / eps
                        dFaa = K[i]* K[j]* K[k]* dk**3* dmu* dphi* dBda[i,j,k,l,m]**2 / Covs[i,j,k,l,m]
                        if not np.isnan(dFaa):
                            # print('NaN for k1,k2,k3 = ', K[i], K[j], K[k])
                            faa += dFaa
                        else:
                            notTriangles += 1
    # print('Number of non triangles: ', notTriangles)
    return faa

mu = np.linspace(-1, 1, 2)
phi = np.linspace(0, 1, 2)
apar = 1.01
aper = 0.99
f = 0.4
b1 = 1.7
b2 = 0
navg = 0.01
Vs = 1
eps = 1e-6

# var = (k1, k2, k3, mu, phi)
parc = (apar, aper, f, b1, b2)

Kmax, Kmin = (0.3, 1e-3)
Faa = []
for Nk in range(10, 50, 10):
    dk = (Kmax - Kmin) / Nk
    K = np.arange(Kmin, Kmax, dk)
    Faa.append(0)

    Faa[-1] = compFaa(K, mu, phi, Nk, len(mu), len(phi), apar, aper, f, b1, b2, navg, Vs)

    print(Faa[-1]) 

# plt.plot(Ks, B1s, label='Bispectrum'); plt.xlabel('k1');plt.ylabel('Bisp(k1, 0.25, 0.25,pi/15,pi/15)'), plt.legend()
# plt.show()
# plt.plot(Ks, Covs, label='Covariance'); plt.xlabel('k1');plt.ylabel('CovB(k1, 0.25, 0.25,pi/15,pi/15)'), plt.legend()
# plt.show()
# plt.plot(Ks, B1s/Covs, label='SNR'); plt.xlabel('k1');plt.ylabel('snr(k1, 0.25, 0.25,pi/15,pi/15)');plt.legend()
# plt.show()


#k1k2k3*dK**3*dmu*dphi
