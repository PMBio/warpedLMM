import pysnptools.util.pheno 
from pysnptools.snpreader.bed import Bed
import pysnptools.util as srutil
import numpy as np
import pandas

def load_data(snp_file, pheno_file, covar_file):
    # Load SNP data
    snp_reader = Bed(snp_file)

    # Load phenotype
    pheno = pysnptools.util.pheno.loadPhen(pheno_file)

    # Load covariates
    if covar_file is not None:
        covar = pysnptools.util.pheno.loadPhen(covar_file)
        snp_reader, pheno, covar = srutil.intersect_apply([snp_reader, pheno, covar])
        covar = covar['vals']
    else:
        snp_reader, pheno = srutil.intersect_apply([snp_reader, pheno])
        covar = None

    snp_data = snp_reader.read().standardize()
    Y = pheno['vals']
    Y -= Y.mean(0)
    Y /= Y.std(0)

    X = 1./np.sqrt((snp_data.val**2).sum() / float(snp_data.iid_count)) * snp_data.val
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
