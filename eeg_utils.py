import numpy as np
import sys

class MultivariateGaussianLikelihood:
    def __init__(self):
        self.mu = None
        self.Sigma = None
        self.isFit = 0

    def fit(self, data):
        self.mu = np.mean(data, axis=0)
        self.Sigma = np.cov(data.transpose())
        self.isFit=1

    def gaussian(self, pos):
        """adopted from
        https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/"""
        """Return the multivariate Gaussian distribution on array pos.

        pos is an (n x 2) array first dim is n_samples and second 
        is dim_input which is 2. Need to check whether it generalize to
        bigger than dim_input is 2.
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.
        """
        if self.isFit == 0:
            print("MultivariateGuassianLikelihood is no fit")
            sys.exit(-1)

        n = self.mu.shape[0]
        Sigma_det = np.linalg.det(self.Sigma)
        Sigma_inv = np.linalg.inv(self.Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-self.mu, Sigma_inv, pos-self.mu)

        return np.exp(-fac / 2) / N

def smoothBySlidingWindow(data, window_size):
    '''
    write IO info
    '''
    data_len = data.shape[0]
    wsz = window_size
    l = []

    for pi in range(wsz-1, data_len):
        data_range = range(pi-wsz+1, pi+1)
        w = in_nums[data_range, :] # window

        a = np.mean(w, axis=0)
        l.append(a)

    pad = np.empty((wsz-1, in_n)); pad[:] = np.nan
    res = np.vstack([pad] + l)

    return res

if __name__=='__main__':
    data = np.loadtxt('./data/eeg_tr.csv', delimiter=',')

    glf = MultivariateGaussianLikelihood() # gaussian likelihood function

    glf.fit(data)

    pos = data[:10] # random example

    print(glf.gaussian(pos))

    for i in range(10):
        print(glf.gaussian(data[i]))

    import pdb; pdb.set_trace()
