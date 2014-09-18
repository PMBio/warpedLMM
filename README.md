# WarpedLMM

#### A python package implementing warped linear mixed models.

Genome-wide association studies, now routine, still have many remaining methodological open problems. Among the most successful models for GWAS are linear mixed models, also used in several other key areas of genetics, such as phenotype prediction and estimation of heritability. However, one of the fundamental assumptions of these models-that the data have a particular distribution (i.e., the noise is Gaussian-distributed)-rarely holds in practice. As a result, standard approaches yield sub-optimal performance, resulting in significant losses in power for GWAS, increased bias in heritability estimation, and reduced accuracy for phenotype predictions.

This repository contains a python implementation of the warped linear mixed model. WarpedLMM automatically learns an optimal "warping function" for the phenotype simultaneously as it models the data. Our approach effectively searches through an infinite set of transformations, using the principles of statistical inference to determine an optimal one. 

#### Installation

WarpedLMM is available from the python package index.

```shell
pip install warpedlmm
```

#### Getting started

* [Paper] http://dx.doi.org/10.1038/ncomms5890

User documentation is coming soon.

#### Citing WarpedLMM

Please cite WarpedLMM in your publications if it helps your research:

    @article{fusi2014genetic,
             title={Warped linear mixed models for the genetic analysis of transformed phenotypes.},
             author={Fusi, Nicolo and Lippert, Christoph and Lawrence, Neil D and Stegle, Oliver},
             journal={Nature Communications (in press)},
             doi={10.1038/ncomms5890},
             year={2014}}               

#### Contacting us 

You can submit bug reports using the github issue tracker. 
If you have any other question please contact: fusi [at] microsoft com, stegle [at] ebi ac uk