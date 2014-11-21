import fastlmm.pyplink.plink as plink
from pysnptools.pysnptools.snpreader.bed import Bed
import pysnptools.pysnptools.util.util as srutil
import numpy as np
import pandas

def load_data(snp_file, pheno_file, covar_file):
    # Load SNP data
    snp_data = Bed(snp_file)
    snp_data = snp_data.read().standardize()

    # Load phenotype
    pheno = plink.loadPhen(pheno_file)

    # Load covariates
    if covar_file is not None:
        covar = plink.loadPhen(covar_file)
        snp_data, pheno, covar = srutil.intersect_apply([snp_data, pheno, covar])
        covar = covar['vals']
    else:
        snp_data, pheno = srutil.intersect_apply([snp_data, pheno])
        covar = None

    Y = pheno['vals']
    Y -= Y.mean(0)
    Y /= Y.std(0)

    X = 1./np.sqrt((snp_data.val**2).sum() / float(snp_data.val.shape[0])) * snp_data.val
    K = np.dot(X, X.T) # TODO use symmetric dot to speed this up

    assert np.all(pheno['iid'] == snp_data.iid), "the samples are not sorted"

    return snp_data, pheno, covar, X, Y, K

def write_results_to_file(snp_data, pv, results_filename):
    results = pandas.DataFrame(index=snp_data.sid, columns=['chromosome', 'genetic distance', 'position', 'p-value'])

    results['chromosome'] = snp_data.pos[:, 0]
    results['genetic distance'] = snp_data.pos[:, 1]
    results['position'] = snp_data.pos[:, 2]
    results['p-value'] = pv[:, None]

    assert np.all(results.index == snp_data.sid) and np.all(results['p-value'] == pv), "the pvalues and/or SNP ids are not in order in the output file"

    results.to_csv(results_filename)
