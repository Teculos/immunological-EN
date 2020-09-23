# immunological-EN
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3885868.svg)](https://doi.org/10.5281/zenodo.3885868)
[![license](https://img.shields.io/github/license/DAVFoundation/captain-n3m0.svg?style=flat-square)](https://github.com/Teculos/immunological-EN/blob/master/LICENSE)


# Package Information and Contact
* Code Tested in: R version 3.6.3 on Ubuntu 18.04.4 LTS
* Prepared by: Tony Culos (tculos@stanford.edu)


# Summary
We introduce the immunological Elastic Net (iEN) which integrates mechanistic immunological knowledge into a machine learning framework. Here we provide code for the application of iEN models and its optimization given a set of hyperparameter values. For a more comprehensive description of this method please see `Integration of Mechanistic Immunological Knowledge into a Machine Learning Pipeline Improves Predictions`.

# Package Installation
Installation of the 'immunological-EN' can be accomplished easiest through the terminal. All libraries dependent for the optimization and fitting of iEN models must be installed prior to building and installing the package from the source files. To install all dependencies please run this command prior to installation ```install.packages(c('pROC', 'Metrics', 'Matrix', 'glmnet', 'knitr'))```

See `DESCRIPTION` file for a full list of imported and suggested packages.

## Installation From .tar.gz
1. Download the entire repository
1. Run ```install.packages(path_to_file, repos = NULL, type="source")``` where the file is ```iEN_0.99.0.tar.gz```
1. ```iEN``` package should now be available in R via the ```library('iEN')``` command

## Build .tar.gz From Source Files
1. Download the entire repository and remove ```iEN_0.99.0.tar.gz``` file
1. In the terminal navigate to the previously mentioned folder location and run the following command
```R CMD Build immunological-EN-master```
If different, adapt this commond to accomadate whichever folder name was used
1. Next install the `.tar.gz` file which was built
```R CMD INSTALL iEN_0.99.0.tar.gz```
