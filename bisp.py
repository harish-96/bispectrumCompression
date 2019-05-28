import matplotlib.pyplot as plt
from numba import jit
from BFisherutils import *
import pdb

def KL_matrix(mean_der, cov):
    return np.dot(mean_der.T, np.linalg.inv(cov))

def Faa(K, mu, phi, parc, pars, par_indices=[0]):
    dk = K[1] - K[0]
    dmu = mu[1] - mu[0]
    dphi = phi[1] - phi[0]
    eps = 1e-6

    dBda = []

    Bs = Bisp_vec((K, K, K, mu, phi), parc, pars)
    Covs = CovB_vec((K, K, K, mu, phi), parc, pars)
    for i, par_index in enumerate(par_indices):
        dparams = np.zeros_like(parc, dtype=np.float64)
        dparams[par_index] = eps
        dBda.append((Bisp_vec((K, K, K, mu, phi), parc+dparams, pars) - Bs)/eps)

    k1 = K.reshape(len(K),1,1,1,1) * np.ones((Bs.shape))
    k2 = K.reshape(1,len(K),1,1,1) * np.ones((Bs.shape))
    k3 = K.reshape(1,1,len(K),1,1) * np.ones((Bs.shape))

    Covs /= (k1*k2*k3 * dk**3 * dmu * dphi)

    Bs = Bs[~np.isnan(Bs)]
    Covs = Covs[~np.isnan(Covs)]
    dBda_flat = np.zeros((len(Covs),len(par_indices)))
    for i,it in enumerate(dBda):
        dBda_flat[:,i] = it[~np.isnan(it)]

    faa = np.dot(dBda_flat.T, np.dot(np.diag(1/Covs), dBda_flat))

    return faa

def Faa_KL(K, mu, phi, parc, pars, par_indices=[0]):
    dk = K[1] - K[0]
    dmu = mu[1] - mu[0]
    dphi = phi[1] - phi[0]
    eps = 1e-6

    dBda = []

    Bs = Bisp_vec((K, K, K, mu, phi), parc, pars)
    Covs = CovB_vec((K, K, K, mu, phi), parc, pars) 
    for i, par_index in enumerate(par_indices):
        dparams = np.zeros_like(parc, dtype=np.float64)
        dparams[par_index] = eps
        dBda.append((Bisp_vec((K, K, K, mu, phi), parc+dparams, pars) - Bs)/eps)

    k1 = K.reshape(len(K),1,1,1,1) * np.ones(Bs.shape)
    k2 = K.reshape(1,len(K),1,1,1) * np.ones(Bs.shape)
    k3 = K.reshape(1,1,len(K),1,1) * np.ones(Bs.shape)
    Covs /= (k1*k2*k3 * dk**3 * dmu * dphi)

    Bs = Bs[~np.isnan(Bs)]
    Covs = Covs[~np.isnan(Covs)]

    dBda_flat = np.zeros((len(Covs),len(par_indices)))
    for i,it in enumerate(dBda):
        dBda_flat[:,i] = it[~np.isnan(it)]

    A = np.dot(dBda_flat.T, np.diag(1/Covs))

    f_t = np.linalg.inv(np.dot(A, np.dot(np.diag(Covs), A.T)))
    faa_kl = np.dot(dBda_flat.T, np.dot(A.T, np.dot(f_t, np.dot(A, dBda_flat))))

    pdb.set_trace()

    return faa_kl


mu = np.linspace(-1, 1, 2)
phi = np.linspace(0, np.pi, 2)
apar = 1.01
aper = 0.99
f = 0.4
b1 = 1.7
b2 = 0
navg = 0.01
Vs = 1
eps = 1e-6

parc = (apar, aper, f, b1, b2)
pars = (navg, Vs)

Kmax, Kmin = (0.19, 1e-3)
faa, dk = [], []
for Nk in range(5, 30, 5):
    dk.append((Kmax - Kmin) / Nk)
    faa.append(0)

    K = np.arange(Kmin, Kmax, dk[-1])

    faa[-1] = Faa(K, mu, phi, parc, pars, [0,1,2])

    print(faa[-1]) 

# plt.plot(dk, Faa, '*-')
# plt.xlabel('$\Delta k$')
# plt.ylabel('$F_{aa}$')
# plt.show()

# plt.plot(Ks, B1s, label='Bispectrum'); plt.xlabel('k1');plt.ylabel('Bisp(k1, 0.25, 0.25,pi/15,pi/15)'), plt.legend()
# plt.show()
# plt.plot(Ks, Covs, label='Covariance'); plt.xlabel('k1');plt.ylabel('CovB(k1, 0.25, 0.25,pi/15,pi/15)'), plt.legend()
# plt.show()
# plt.plot(Ks, B1s/Covs, label='SNR'); plt.xlabel('k1');plt.ylabel('snr(k1, 0.25, 0.25,pi/15,pi/15)');plt.legend()
# plt.show()


#k1k2k3*dK**3*dmu*dphi
