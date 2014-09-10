import warpedlmm
import argparse
import testing
from numpy.testing import Tester
import pandas
import pysnptools.snpreader.bed
import warpedlmm.fastlmm

if __name__ == '__main__':
    usage = 'usage: warpedlmm snp_file phenotype_file'
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('snp_file', help='file containing the SNP data (only .bed PLINK files are supported for now)')
    parser.add_argument('phenotype_file',  help='phenotype file in csv format (one sample per row)')
    parser.add_argument('--test', dest='run_unit_tests', action='store_true', default=False, help='run unit tests (default: False)')
    parser.add_argument('--covariates', dest='covariates', action='store', type=str, default=None, help='covariates file (optional)')
    parser.add_argument('--save', dest='save', action='store_true', default=False, help='save transformed phenotype to file. A "_transformed" is appended to the original phenotype filename. (default: False)')
    parser.add_argument('--random_restarts', dest='random_restarts', action='store', default=3, type=int, help='number of random restarts')
    parser.add_argument('--qvalue_cutoff', dest='qv_cutoff', action='store', default=None, type=float, help='q-value cutoff for inclusion of large effect loci in the model (by default the model uses a p-value cutoff at 5e-8, see --pvalue_cutoff)')
    parser.add_argument('--pvalue_cutoff', dest='pv_cutoff', action='store', default=None, type=float, help='p-value cutoff for inclusion of large effect loci in the model (by default 5e-8)')
    parser.add_argument('--max_covariates', dest='max_covariates', action='store', default=None, type=int, help='maximum number of SNPs that can be included in the model (default: 10)')


    options = parser.parse_args()

    if options.run_unit_tests:
        Tester(testing).test(verbose=-1)

    # Load SNP data
    snp_data = pysnptools.snpreader.bed.Bed(options.snp_file)
    snp_data.read()

    # Load phenotype
    pheno_data = np.loadtxt(options.phenotype_file, delimiter=',', fmt='%s')

    # Load covariates
    if options.covariates is not None:
        covariates_data = np.loadtxt(options.covariates, delimiter=',', fmt='%s')

    X = None # TODO: standardize SNPs
    K = None # TODO compute kernel
    y_pheno, _, _, estimated_h2 = warpedlmm.stepwise(Y, X, K, covariates=covariates_data,
                                                     max_covariates=options.max_covariates, num_restarts=options.random_restarts,
                                                     qv_cutoff=options.qv_cutoff,
                                                     pv_cutoff=options.pv_cutoff)

    pv, h2 = warpedlmm.fastlmm.assoc_scan(y_pheno.copy(), X, covariates=covariates, K=K)
    # TODO write to file
    
    if options.save:
        pheno_file_name = options.phenotype_file.replace('.txt', '')
        # TODO write to file attaching family ids and so on
