import numpy as np
from warpedlmm.warpedlmm import WarpedLMM
from warpedlmm.stepwise import warped_stepwise
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

    def test_heritability(self):
        snp_data, pheno, covar, X, Y, K = util.load_data(os.path.dirname(os.path.realpath(__file__))+"/sim_data", os.path.dirname(os.path.realpath(__file__))+"/sim_data.pheno", None)
        y_pheno, m, _, estimated_h2 = warped_stepwise(Y, X, K, covariates=covar,
                                                               max_covariates=10,
                                                               num_restarts=1,
                                                               qv_cutoff=0.05,
                                                               pv_cutoff=None)
        print estimated_h2
        self.assertTrue(np.allclose(estimated_h2, 0.5, atol=5e-2))

    def test_stepwise(self):
        '''
        test for bug #2 (pmbio/warpedlmm)
        '''
        
        snp_data, pheno, covar, X, Y, K = util.load_data(os.path.dirname(os.path.realpath(__file__))+"/sim_data", os.path.dirname(os.path.realpath(__file__))+"/sim_data.pheno", None)
        y_pheno, m, _, estimated_h2 = warped_stepwise(Y, X, K, covariates=covar,
                                                               max_covariates=2,
                                                               num_restarts=1,
                                                               qv_cutoff=0.5,
                                                               pv_cutoff=None)
        print estimated_h2
        self.assertTrue(np.allclose(estimated_h2, 0.5, atol=5e-2))

class LoaderTests(unittest.TestCase):
    def test_load(self):
        snp_data, pheno, covar, X, Y, K = util.load_data(os.path.dirname(os.path.realpath(__file__))+"/sim_data", os.path.dirname(os.path.realpath(__file__))+"/sim_data.pheno", None)
