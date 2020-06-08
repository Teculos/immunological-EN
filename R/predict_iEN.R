#' predicts with a iEN object, defualt of the model is to predict the new data with the mean coefficient model over each out of sample instance. With the default phi applied to to newx being the mean phi from optimal models found during CV.
#'
#'
#' @param object object of type iEN
#' @param newx new data to predict
#' @param priors priors indicating how phi will be applied, should be identical to the priors used during cross-validation.
#' @param phi The amount of prioritization to use on newx, this should be the same phi as is used to construct the iEN model during prediction. Default for this method is to apply the mean phi from out of sample models.
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
#' predict(model, X, priors)
#' @return vector of predicted values using object to predict newx
#'
#' @export
setMethod("predict", 
    signature(object = "iEN"),
    function(object, newx, priors, phi){
        
        if( !identical(names(priors),colnames(newx)) ){
            stop("names of priors does not match column names on newx.")
        }

        if(!missing(phi)){
            newx <- newx %*% diag(exp(-(1-priors) * phi))
        }else{
            newx <- newx %*% diag(exp(-(1-priors) * mean(object@opt.params[,3])))
        }
        model <- colMeans(as.matrix(object@betas))

        return(newx %*% model[-1] + model[1])
    })