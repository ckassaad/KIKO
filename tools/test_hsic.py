from tools.hsic import HSIC
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import gamma


class TestHSIC:
    def __init__(self, method='gamma', kernel='rbf', gamma_param='median', n_iter=0, num_processor=3):
        '''
        :param method: method test used to calculate the p-value: gamma test or permutation test
        :param kernel: rbf or laplace
        :param gamma_param: gamma of the kernel, by default it uses the median heuristic
        :param n_iter: number of permutations (it is used only when method=permutation)
        :param num_processor: number of processors used
        '''
        self.method = method
        hsic = HSIC(kernel, gamma_param)
        self.hsic = hsic
        self.n_iter = n_iter
        self.init_hsic = 0
        self.num_processor = num_processor

    # calculating hsic after applying permutations of y
    def permute(self, i, x, y):
        '''
        :param i: iteration number
        :param x: data x
        :param y: data y
        :return: hsic after a permutation
        '''
        np.random.seed(i)
        idx_permutation = np.random.permutation(y.shape[0])
        y_temp = y[idx_permutation, :]
        res = self.hsic.fit(x, y_temp)
        return res

    # calulation of the p-value
    def fit(self, x, y):
        '''
        :param x: data x
        :param y: data y
        :return: p-value
        '''
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        init_hsic = self.hsic.fit(x, y)
        self.init_hsic = init_hsic

        if self.method == 'permutation':

            perm_hsic = Parallel(n_jobs=self.num_processor)(delayed(self.permute)(i, x, y) for i in range(self.n_iter))

            p_value = (np.sum(perm_hsic >= init_hsic)+1)/(self.n_iter+1)

        elif self.method == 'gamma':
            m = x.shape[0]
            #if dhsic is true we use the same implementation as dhsic library (R),
            dhsic = True
            if not dhsic:
                mux = (np.sum(self.hsic.k1) - np.diag(self.hsic.k1).sum())/ (m) / (m - 1)
                muy = (np.sum(self.hsic.k2) - np.diag(self.hsic.k2).sum())/ (m) / (m - 1)
                E = (1 + mux * muy - mux - muy) / m
                print(E)

                h = np.identity(m) - 1 / m
                k1 = np.matmul(np.matmul(h, self.hsic.k1), h)
                k2 = np.matmul(np.matmul(h, self.hsic.k2), h)
                V = (1 / 6 * k1 * k2) ** 2
                V = 1 / m / (m - 1) * (np.sum(V) - sum(np.diag(V)))
                V = 72 * (m - 4) * (m - 5) / m / (m - 1) / (m - 2) / (m - 3) * V
                print(V)
            else:
                a = [np.sum(K)/(m**2) for K in [self.hsic.k1, self.hsic.k2]]
                b = [np.sum(K**2)/(m**2) for K in [self.hsic.k1, self.hsic.k2]]
                c = [np.sum(np.sum(K, axis=0)**2)/(m**3) for K in [self.hsic.k1, self.hsic.k2]]
                d = [i**2 for i in a]

                prod_a = np.prod(a)
                prod_b = np.prod(b)
                prod_c = np.prod(c)
                prod_d = np.array(prod_a**2)

                outprod_a = prod_a / a
                outprod_b = prod_b / b
                outprod_c = prod_c / c
                outprod_d = outprod_a ** 2

                # expectation
                E = (1 + prod_a - np.sum(outprod_a)) / m

                term1 = prod_b
                term2 = prod_d
                term3 = 2 * prod_c
                term4 = 0
                term5 = 0
                term6 = 0
                term7 = 0

                term4 = term4 + b[0] * outprod_d[0]
                term5 = term5 + b[0] * outprod_c[0]
                term6 = term6 + c[0] * outprod_d[0]
                term7 = term7 + 2 * c[0] * c[1] * outprod_d[0] / d[1]

                term4 = term4 + b[1] * outprod_d[1]
                term5 = -2 * (term5 + b[1] * outprod_c[1])
                term6 = -2 * (term6 + c[1] * outprod_d[1])

                m4 = m*(m - 1)*(m - 2) * (m - 3)
                # variance
                V = (2*(m-4) * (m - 5)/m4)*(term1+term2+term3+term4+term5+term6+term7)

            # calculation of alpha and beta of the gamma approximation
            a = (E ** 2) / V
            b = (m * V) / E

            # test statistic
            test_stat = init_hsic*m

            # Cumulative distribution function²
            p_value = 1-gamma.cdf(test_stat, a, scale=b)

        else:
            print('Unavailable method')

        return p_value


if __name__ == "__main__":

    np.random.seed(1)
    import time

    start = time.time()

    X =  np.array([0.136486655,	  0.108931511,	  0.022105488,	  0.349472863,	  0.148249433,	 -0.321564702,	  0.320629400,	 -0.282624440,	  0.263522936,	 -0.316252276])
    Y =  np.array([ -0.1669332713,	  0.4886635816,	  0.5315530519,	  0.1330376544,	 -0.0632027887,	  0.1640341743,	 -0.1833757726,	  0.3754725901,	 -0.0722728821,	 -0.0395241960])

    model = TestHSIC(method='gamma', n_iter=100000)
    p = model.fit(X, Y)
    print(p)
    print(model.init_hsic)
    end = time.time()
    print('time: ' + str(end - start))

    # ------------------------------
    #Using dhsic of R:
    # library(dHSIC)
    # X = c(0.136486655, 0.108931511, 0.022105488, 0.349472863, 0.148249433, -0.321564702, 0.320629400, -0.282624440,
    #       0.263522936, -0.316252276)
    # Y = c(-0.1669332713, 0.4886635816, 0.5315530519, 0.1330376544, -0.0632027887, 0.1640341743, -0.1833757726,
    #       0.3754725901, -0.0722728821, -0.0395241960)
    # dhsic.test(X, Y, method="gamma")