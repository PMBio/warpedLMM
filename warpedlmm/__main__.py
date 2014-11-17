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


import warpedlmm
import argparse
import testing
import pandas
from numpy.testing import Tester
import util
import fastlmm_interface as fastlmm
import numpy as np
import stepwise
import matplotlib.pyplot as plt

if __name__ == '__main__':
    usage = 'usage: warpedlmm snp_file phenotype_file'
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('snp_file', help='file containing the SNP data (only .bed PLINK files are supported for now)')
    parser.add_argument('phenotype_file',  help='phenotype file in csv format (one sample per row)')
    parser.add_argument('--test', dest='run_test', action='store_true', default=False, help='check that the warping is working by artificially transforming the phenotype (it computes exp(phenotype)) (default: False)')
    parser.add_argument('--covariates', dest='covariates', action='store', type=str, default=None, help='covariates file (optional)')
    parser.add_argument('--save', dest='save', action='store_true', default=False, help='save transformed phenotype to file. A "_WarpedLMM" is appended to the original phenotype filename. (default: False)')
    parser.add_argument('--random_restarts', dest='random_restarts', action='store', default=3, type=int, help='number of random restarts')
    parser.add_argument('--qvalue_cutoff', dest='qv_cutoff', action='store', default=None, type=float, help='q-value cutoff for inclusion of large effect loci in the model (by default the model uses a p-value cutoff at 5e-8, see --pvalue_cutoff)')
    parser.add_argument('--pvalue_cutoff', dest='pv_cutoff', action='store', default=5e-8, type=float, help='p-value cutoff for inclusion of large effect loci in the model (by default 5e-8)')
    parser.add_argument('--max_covariates', dest='max_covariates', action='store', default=10, type=int, help='maximum number of SNPs that can be included in the model (default: 10)')
    parser.add_argument('--output_directory', dest='out_dir', action='store', default=None, type=str, help='output directory (default: same directory as the phenotype)')
    parser.add_argument('--normal-scan-first', dest='normal', action='store_true', default=False, help='run an association scan without warping')

    
    options = parser.parse_args()

    snp_data, pheno, covar, X, Y, K = util.load_data(options.snp_file, options.phenotype_file, options.covariates)

    if options.run_test:
        Y = np.exp(Y)

    y_pheno, m, _, estimated_h2 = stepwise.warped_stepwise(Y, X, K, covariates=covar,
                                                           max_covariates=options.max_covariates, num_restarts=options.random_restarts,
                                                           qv_cutoff=options.qv_cutoff,
                                                           pv_cutoff=options.pv_cutoff)

    pv, h2 = fastlmm.assoc_scan(y_pheno.copy(), X, covariates=covar, K=K)

    
    if options.out_dir is None:
        results_file_name = options.phenotype_file.replace('.txt', '')
        results_file_name += "_warpedlmm_results.txt"
    else:
        results_file_name = options.out_dir + "/warpedlmm_results.txt"

    util.write_results_to_file(snp_data, pv, results_file_name)


    if options.normal:
        pv_base, h2_base = fastlmm.assoc_scan(Y.copy(), X, covariates=covar, K=K)
        util.write_results_to_file(snp_data, pv_base, results_file_name.replace('warpedlmm', 'fastlmm'))

    if options.save:
        if options.out_dir is None:
            pheno_file_name = options.phenotype_file.replace('.txt', '')
            pheno_file_name += "_warpedlmm_pheno.txt"
            trafo_file_name = options.phenotype_file.replace('.txt', '')
            trafo_file_name += "_warpedlmm_transformation.png"
        else:
            pheno_file_name = options.out_dir + "/warpedlmm_pheno.txt"
            trafo_file_name = options.out_dir + "/warpedlmm_transformation.png"

        np.savetxt(pheno_file_name, np.concatenate((np.array(snp_data.iid, dtype='|S15'), y_pheno), axis=1), fmt='%s')
        m.plot_warping()
        plt.savefig(trafo_file_name)
