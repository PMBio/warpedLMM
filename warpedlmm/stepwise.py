# Copyright 2014 Nicolo Fusi

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from warpedlmm import WarpedLMM
import util.qvalue as qvalue
import fastlmm

def warped_stepwise(Y, X=None, K=None, covariates=None, num_restarts=1, max_covariates=10, qv_cutoff=None, pv_cutoff=5e-8):
    Xt = X.copy()
    Xt -= Xt.mean(axis=0)
    Xt /= Xt.std(axis=0)
    Kt = np.dot(Xt, Xt.T)

    if covariates is None:
        num_covariates = 0
    else:
        num_covariates  = covariates.shape[1]

    included = []
    converged = False
    iterations = 1
    estimated_h2s = []
    likelihoods = []

    while not converged:
        m = WarpedLMM(Y, X, K=K, X_selected=covariates, warping_terms=3)
        m.optimize_restarts(num_restarts, messages=1)
        y_pheno = m.Y.copy()

        y_pheno -= y_pheno.mean()
        y_pheno /= y_pheno.std()

        if covariates is not None:
            covariates -= covariates.mean(0)
            covariates /= covariates.std(0)

        #import panama.core.testing as testing
        # pv_lmm_panama = testing.interface(X.copy(), y_pheno.copy(), K.copy(), covariates)[0].flatten()
        pv_lmm, h2 = fastlmm.assoc_scan(y_pheno.copy(), X.copy(), K=K.copy(), covariates=covariates)

        if qv_cutoff is not None:
            qv_lmm = qvalue.estimate(pv_lmm)
            sorted_qv_ind = np.argsort(qv_lmm)
            candidate_index = sorted_qv_ind[0]
            candidate_sign = qv_lmm[candidate_index]
            significant = (qv_lmm <= qv_cutoff)
            cutoff = qv_cutoff
        else:
            significant = (pv_lmm <= pv_cutoff)
            sorted_pv_ind = np.argsort(pv_lmm)
            candidate_index = sorted_pv_ind[0]
            candidate_sign = pv_lmm[candidate_index]
            cutoff = pv_cutoff
            
        likelihoods.append(m.log_likelihood())
        # h2 = m.params['sigma_g']/m.params['sigma_e']
        estimated_h2s.append(h2)

        status = "Iteration: {0}, significant SNPs: {1}, included SNPs: {2},  heritability: {3:.4f}, f: {4}".format(iterations, significant.sum(), len(included), estimated_h2s[-1], likelihoods[-1])
        print status

        if candidate_sign > cutoff or len(included) >= max_covariates:
            converged = True
            continue

        X_sign = Xt[:, candidate_index:candidate_index+1]
        if covariates != None:
            covariates = np.append(covariates, X_sign, axis=1)
        else:
            covariates = X_sign

        iterations += 1
        included.append(candidate_index)

    return y_pheno, m, included, estimated_h2s[-1]


if __name__ == '__main__':
    np.random.seed(1)
    X = np.random.randn(500, 100)
    y = np.dot(X, np.random.randn(100, 1)) + np.random.randn(500, 1)

    warped_stepwise(y, X=X)
