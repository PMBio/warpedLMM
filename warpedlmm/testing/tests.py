import numpy as np
from warpedlmm.warpedlmm import WarpedLMM
import warpedlmm.util as util
import unittest
import scipy as sp
import os

class ModelTests(unittest.TestCase):
    def test_gradients(self):
        N = 120
        X = np.random.randn(N, 1)
        X -= X.mean()
        X /= X.std()
        Z = X * 0.8 + np.random.randn(N, 1)*0.2
        Z += np.abs(Z.min()) + 0.5
        Y = np.log(Z)
        m = WarpedLMM(Y, X, warping_terms=2)
        for i in range(10):
            m.randomize()
            self.assertTrue(sp.optimize.check_grad(m._f, m._f_prime, m._get_params()) < 1e-4)

    def test_model(self):
        N = 120
        X = np.random.randn(N, 1)
        X -= X.mean()
        X /= X.std()
        Z = X * 0.8 + np.random.randn(N, 1)*0.2
        Z += np.abs(Z.min()) + 0.5
        Y = np.log(Z)#Z**(1/3.0)
        m = WarpedLMM(Y, X, warping_terms=2)
        m.randomize()
        m.optimize(messages=0)

        self.assertTrue(sp.stats.pearsonr(m.Y, Z)[0] >= 0.9)

class LoaderTests(unittest.TestCase):
    def test_load(self):
        snp_data, pheno, covar, X, Y, K = util.load_data(os.path.dirname(os.path.realpath(__file__))+"/sim_data", os.path.dirname(os.path.realpath(__file__))+"/sim_data.pheno", None)
