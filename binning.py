import matplotlib.pyplot as plt     
from BFisherutils import *
import pdb


def compFaa(K, mu, Nk, Nmu, apar, aper, f, b1, b2, navg, Vs, bin_size=8):
    Nk = Nk - Nk%bin_size
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

    Nk_t = Nk//bin_size
    A = conv1d_matrix((Nk_t, Nk), np.ones(bin_size)/bin_size)
    K_t = np.dot(A, K)
    dk_t = K_t[1] - K_t[0]
    CovP_t = np.zeros((Nk_t, Nk_t, Nmu))
    Fp_t = np.zeros((Nk_t, Nk_t, Nmu))
    dP_tda = np.zeros((Nk//bin_size, Nmu))

    tmp_f = np.zeros(Nk)
    for i in range(Nk):
        for l in range(Nmu):
            Ps[i, l] = Pk((K[i], mu[l]), (apar, aper, f, b1, b2))
            Covs[i,l] = CovP((K[i], mu[l]), (apar, aper, f, b1, b2), (navg, Vs))
            dPda[i,l] = (Pk((K[i], mu[l]), (apar+eps, aper, f, b1, b2)) - Ps[i,l]) / eps
            faa += K[i]**2 * dPda[i,l]**2 / Covs[i,l]

    # Computing the same quantities for binned data
    for i in range(Nmu):
        dP_tda[:, i] = np.dot(A, dPda[:,i])
        CovP_t[:, :, i] = np.dot(A, np.dot(np.diag(Covs[:,i]), A.T))
        Fp_t[:,:,i] = np.linalg.inv(CovP_t[:, :, i])
    
    for i in range(Nk//bin_size):
        for k in range(Nmu):
            faa_t += K_t[i]**2 * dP_tda[i, k]**2 * Fp_t[i, i, k]
            # faa_t += 2*np.pi*K_t[i]**2*K_t[j]**2*dk_t**2*dmu*dP_tda[i, k] * dP_tda[j, k] * Fp_t[i, j, k]
    # pdb.set_trace()

    # return Nk * faa, Nk_t * faa_t
    return faa, faa_t

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

parc = (apar, aper, f, b1, b2)

Nks = np.arange(16, 80, 8)
bin_sizes = np.array([2, 4, 8])
dk = (Kmax - Kmin) / Nks

Faa = np.zeros(len(Nks))
Faa_t = np.zeros((len(Nks), len(bin_sizes)))

for i in range(len(Nks)):
    for j in range(len(bin_sizes)):
        K = np.arange(Kmin, Kmax, dk[i])

        Faa[i], Faa_t[i, j] = compFaa(K, mu, Nks[i], len(mu), apar, aper, f, b1, b2, navg, Vs, bin_sizes[j])

        print(Faa[i], Faa_t[i, j]) 

plt.plot(dk, Faa, '*-', label='$F_{aa}$')
for i in range(len(bin_sizes)):
    plt.plot(dk, Faa_t[:, i], '*-', label=r'$\tildeF_{aa}$, bin size='+str(bin_sizes[i]))
plt.xlabel('$\Delta k$')
plt.ylabel('$F_{aa}$')
plt.legend()
plt.savefig('./plot_bins.png')
plt.show()

