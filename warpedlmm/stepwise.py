import numpy as np
from warpedlmm import WarpedLMM
import util.qvalue as qvalue
import fastlmm

def warped_stepwise(Y, X=None, K=None, covariates=None, num_restarts=1, max_covariates=10, qv_cutoff=0.05):
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
    cutoff = qv_cutoff
    iterations = 1
    estimated_h2s = []
    likelihoods = []

    while not converged:
        m = WarpedLMM(Y, X, K=K, X_selected=covariates, warping_terms=3)
        m.optimize_restarts(num_restarts, messages=1)
        y_pheno = m.Y.copy()

        y_pheno -= y_pheno.mean()
        y_pheno /= y_pheno.std()

        covariates -= covariates.mean(0)
        covariates /= covariates.std(0)        

        # import panama.core.testing as testing
        # pv_lmm = testing.interface(X.copy(), y_pheno.copy(), K.copy(), covariates)[0].flatten()
        pv_lmm, h2 = fastlmm.assoc_scan(y_pheno.copy(), X.copy(), K=K.copy(), covariates=covariates)
        qv_lmm = qvalue.estimate(pv_lmm)
        sorted_qv_ind = np.argsort(qv_lmm)
        candidate_index = sorted_qv_ind[0]
        candidate_qv = qv_lmm[candidate_index]
        significant = (qv_lmm <= cutoff)

        likelihoods.append(m.log_likelihood())
        h2 = m.params['sigma_g']/m.params['sigma_e']
        estimated_h2s.append(h2)

        status = "Iteration: {0}, significant SNPs: {1}, included SNPs: {2},  heritability: {3:.4f}, f: {4}".format(iterations, significant.sum(), len(included), estimated_h2s[-1], likelihoods[-1])
        print status

        if candidate_qv > cutoff or len(included) >= max_covariates:
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
