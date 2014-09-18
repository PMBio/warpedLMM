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
import fastlmm
import fastlmm.inference.lmm_cov as lmm_cov
import numpy as np
import scipy.stats as st

def assoc_scan(Y, X, covariates=None, K=None):
    lmm = lmm_cov.LMM(X=covariates, Y=Y, G=X, K=K)
    # opt = lmm.find_log_delta(X.shape[1], nGridH2=100, REML=False)
    opt = lmm.findH2(nGridH2=100)
    h2 = opt['h2']
    res = lmm.nLLeval(h2=h2, delta=None, dof=None, scale=1.0, penalty=0.0, snps=X)
    chi2stats = res['beta']*res['beta']/res['variance_beta']
    p_values = st.chi2.sf(chi2stats,1)[:,0]
    return p_values, h2

if __name__ == '__main__':
    X = np.random.randn(500, 100)
    W = np.random.randn(100, 1) * 0.5
    Y = np.dot(X, W) + np.random.randn(500, 1)*0.5

    pv = assoc_scan(Y, X)
