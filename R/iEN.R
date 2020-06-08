#' An S4 class representative of the immunological Elastic Net model
#' 
#' @slot cv.preds out-of-sample predictions made during k-Fold cross validation.
#' @slot cv.eval out-of-sample evaluation of predictive analysis, as defined by cv.kEN parameter eval, this estimates the models ability at predicting unseen data.
#' @slot betas matrix of model coefficients for each optimized out-of-sample model of the K-fold cross validation.
#' @slot dfs vector of values corresponding to the number of non-zero elements in each out-of-sample model
#' @slot nobs number of observations the provided dataset contained.
#' @slot nfeat number of features the provided dataset contained.
#' @slot opt.params optimal parameters (alpha, phi, lambda) for each out-of-sample model generated from the initial search parameters provided.
#' @slot call function call as string from the cv.kEN call which generated this particular object.
#' 
#' @importClassesFrom Matrix dgCMatrix
#' @importClassesFrom methods array list matrix numeric
#' 
#' @export
iEN <- setClass("iEN", slots=list(
    cv.preds = "array",
    cv.eval = "numeric",
    betas = "dgCMatrix",
    dfs = "numeric",
    nobs = "numeric",
    nfeat = "numeric",
    opt.params = "matrix",
    call = "language"))
