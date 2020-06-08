#' Prints iEN object in a formatted way using the knitr package, displays optimized hyper parameters for each out of sample instance during cross-validated gridsearch.
#'
#'
#' @param x object of type iEN
#' @param ... additional parameters fed to print
#' @importFrom knitr kable
#' 
#' 
#' @examples
#' data(test_data)
#' 
#' alphaGrid <- seq(0,1, length.out=2)
#' phiGrid <- exp(seq(log(1),log(10), length.out=2))
#' nlambda <- 3
#' ncores <- 2
#' eval <- "RSS" 
#' family <- "gaussian"
#' intercept <- TRUE
#' standardize <- TRUE
#' center <- TRUE
#' 
#' model <- cv_iEN(X, Y, foldid, alphaGrid, phiGrid, nlambda, NULL, priors, ncores, eval, family, intercept, standardize, center)
#' print(model)
#' @return formatted print
#'
#' @export
setMethod("print", 
    signature(x='iEN'), 
    function(x, ...){
        cat("Estimated model performance, as defined by 'eval' parameter, from out of sample predictions via", paste0(length(x@dfs),"-fold CV was: "), x@cv.eval, "\n")
        print(kable(cbind("features" = x@dfs, alpha=x@opt.params[,1], lambda=x@opt.params[,2], phi=x@opt.params[,3])), ...)
        })