# immunological-EN
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3885868.svg)](https://doi.org/10.5281/zenodo.3885868)

# Package Information and Contact
* Code Tested in: R version 3.6.3 on Ubuntu 18.04.4 LTS
* Prepared by: Tony Culos (tculos@stanford.edu)

# Summary
We introduce the immunological Elastic Net (iEN) which integrates mechanistic immunological knowledge into a machine learning framework. Here we provide code for the application of iEN models and its optimization given a set of hyperparameter values. For a more comprehensive description of this method please see 'Integration of Mechanistic Immunological Knowledge into a Machine Learning Pipeline Improves Predictions'.

# Package Installation
Installation of the 'immunological-EN' can be accomplished easiest through the terminal.

1. Download the entire repository into a folder, we suggest naming this folder 'iEN' or 'immunological-EN' for clarity
1. In the terminal navigate to the previously mentioned folder location and run the following command
'''R CMD Build iEN'''
Adapt this commond to accomadate whichever folder name was used
1. Next install the '.tar.gz' file which was built
'''R CMD INSTALL iEN_0.99.0.tar.gz'''

the iEN model and optimization code will now be available in R through with the library function 'library(iEN)'
