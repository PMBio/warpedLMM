from lmm_cov import LMM as FastLMM
import numpy as np
import scipy.stats as st

def assoc_scan(Y, X, covariates=None, K=None):
    lmm = FastLMM(X=covariates, Y=Y, G=X, K=K)
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
