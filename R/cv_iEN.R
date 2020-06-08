#' Optimizes an iEN model via K-fold cross validation grid search and returns out-of-sample predictions and the associated model meta data.
#'
#'
#' @param X Input matrix of dimensions nobs x nfeat where each row is an observation vector.
#' @param Y Response variable. Is a continuous vector for models of family = "gaussian" and categorical (with two levels) for family = "binomial".
#' @param foldid Vector that identifies which observations belong to which fold during K-fold Cross-Validation. foldid must consist of at least three folds for optimization and model estimation to occur.
#' @param alphaGrid Vector of alpha values for model optimization.
#' @param phiGrid Vector of phi values for model optimization.
#' @param nlambda Lambda values are generated dynamically during cross-validation to avoid any data leak. nlambda determines the number of lambda values to generate.
#' @param lambdas Optional vector for static lambda values.
#' @param priors Continuous values which indicates immune features (columns of X) that are consistent with known biology. Values vary between 0 (low consistency) to 1 (highly consistent) for each immune feature, ie the column space of X.
#' @param ncores Number of cores to use during parallel computing of iEN cross-validation results. For optimal use set ncores = length(alphaGrid) * length(phiGrid).
#' @param eval For binomial models evaluations using Wilcoxon P-value and ROCAUC are provided whereas for Gaussian models RMSE, RSS, Pearson P-value, and Spearman P-value are available.
#' @param family Type of regression model, currently only "Binomial" and "Gaussian" models are supported
#' @param intercept Indicator for inclusion of regresstion intercept (default=TRUE).
#' @param standardize Indication for X variable standardization prior to model fitting (default=TRUE).
#' @param center Indication for X variable centering during scaling (default=TRUE).
#'
#' @importFrom glmnet glmnet predict.glmnet
#' @importFrom Metrics rmse
#' @importFrom methods as new
#' @importFrom pROC roc
#' @importFrom parallel clusterApplyLB clusterExport detectCores makeCluster stopCluster
#' @importFrom stats coefficients cor coef cor.test predict wilcox.test
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
#' @return An object of class "iEN" is returned, which is a class composed of results from the K-fold cross validation and meta data about the analysis. The returned information includes:
#'
#' Out-of-sample predictions from the K-fold cross validation.
#' Evaluation of the out-of-sample predictions as defined by the eval parameter.
#' Coefficients for each out-of-sample regression model, betas.
#' the optimal parameters (alpha, lambda, phi) calculated for each fold of the analysis.
#'
#' @export
cv_iEN  <- function(X, Y, foldid, alphaGrid, phiGrid, nlambda=100, lambdas=NULL, priors, ncores, eval=c("RMSE","RSS","wilcox","ROCAUC","spearman","pearson"), family=c("binomial","gaussian"), intercept=TRUE, standardize=TRUE, center=TRUE){

    #parameter checks
    #todo check that if one of hyper parameters is singular to change that to avoid index issues
    eval <- match.arg(eval)
    family <- match.arg(family)
    if(any(alphaGrid > 1) || any(alphaGrid < 0) || any(phiGrid < 0)){
        stop("Search grid is ill-defined.") 
    }
    if(any(!(sort(names(priors)) %in% colnames(X)))){
        stop("names of priors does not match column names on X.")
    }
    if(ncores > detectCores()){
        stop("Cannot use more cores than provided by system.")
    }
    if(family == "binomial" & !is.factor(Y) & length(levels(Y)) == 2 ){
        stop("With ")
    }
    if((nlambda > 0) && !is.null(lambdas)){
        stop("You have defined both nlambda and lambdas \n please set nlambda < 0 or lambdas = NULL to clarify.")
    }
    if(length(foldid) < nrow(X)){
        stop("foldid does not have corresponding dimensions to X")
    }
    if(length(nlambda) > 1){
        stop("nlambda must be defined as a single integer")
    }

    return(.kfold_prioritization(X, Y, foldid, alphaGrid, phiGrid, nlambda, lambdas, priors, ncores, eval, family, intercept, standardize, center))

}

.kfold_prioritization <- function(X, Y, foldid, alphaGrid, phiGrid, nlambda, lambdas, priors, ncores, eval, family, intercept, standardize, center){

    if(standardize){
            X <- scale(X, center=center)
        if(is.numeric(Y)){
            #uncentered Y such that NO information is shared between folds
            Y <- scale(Y, center=FALSE)
            scale <- attr(Y,"scaled:scale")
        }
    }else{
            scale <- 1
    }

    par.index <- expand.grid(alphaGrid, phiGrid)
    colnames(par.index) <- c("alpha","phi")

    cl <- makeCluster(ncores, type="SOCK")
    clusterExport(cl, list(".gen_lambdas",".iEN", paste0(".",eval,"_eval"),".cv_predict", "glmnet", "roc"), envir=environment())
    res <- clusterApplyLB(cl, fun = .kfold_prioritization_par_Function, seq(nrow(par.index)), par.index=par.index, X=X, Y=Y, nlambda=nlambda,
        lambdas=lambdas, foldid=foldid, eval=eval, family=family, intercept=intercept, priors=priors)
    stopCluster(cl)
    rm(cl)

    eval.grid <- array(data=NA, dim=c(length(unique(foldid)), length(alphaGrid), ifelse(nlambda > 0, nlambda, length(lambdas)), length(phiGrid), 2), 
        dimnames=list(as.character(unique(foldid)), as.character(alphaGrid), NULL, as.character(phiGrid)))

    for(i in seq(nrow(par.index))){
        eval.grid[,as.character(par.index[i,1]),,as.character(par.index[i,2]),1] <- res[[i]][,,1]
        eval.grid[,as.character(par.index[i,1]),,as.character(par.index[i,2]),2] <- res[[i]][,,2]
    }

    max.params <- array(data=NA, dim=c(length(unique(foldid)),3))
    colnames(max.params) <- c("alpha","lambda","phi")
    for(i in seq(length(unique(foldid)))){
        if(max(eval.grid[which(unique(foldid) == i),,,,1], na.rm =TRUE) < 0){
            stop("A solution was not found given the parameters provided.")
        }

        max.index <- arrayInd(which(eval.grid[which(unique(foldid) ==i),,,,1]==max(eval.grid[which(unique(foldid) ==i),,,,1], na.rm = TRUE)), dim(eval.grid[which(unique(foldid) ==i),,,,1])) 
        if(nrow(max.index) > 1){
            select <- sample(seq(nrow(max.index))[-1], 1)
            max.index <- max.index[select,]
        }
        if(length(phiGrid)== 1){
            max.params[which(unique(foldid) ==i),] <- c(alphaGrid[max.index[1]], eval.grid[which(unique(foldid) ==i),max.index[1],max.index[2],,2], phiGrid)
        }else if(length(alphaGrid)==1){
            max.params[which(unique(foldid) ==i),] <- c(alphaGrid[max.index[1]], eval.grid[which(unique(foldid) ==i),max.index[1],,max.index[2],2], phiGrid[max.index[2]])
        }else{  
            max.params[which(unique(foldid) ==i),] <- c(alphaGrid[max.index[1]], eval.grid[which(unique(foldid) ==i),max.index[1],max.index[2],max.index[3],2], phiGrid[max.index[3]])
        }
    }

    oos.pred <- array(data=NA, dim=nrow(X))
    oos.models <- array(data=NA, dim=c(length(unique(foldid)),ncol(X)+1))
    colnames(oos.models) <- c("(Intercept)", ifelse(rep(is.null(colnames(X)), ncol(X)), as.character(seq(ncol(X))), colnames(X)))
    for(i in unique(foldid)){
        temp.X <- X %*% diag(exp(-(1-priors) * max.params[i,3]))

        fit <- .iEN(temp.X[-which(foldid == i),], Y[-which(foldid == i)], max.params[i,1], max.params[i,2], family, intercept)

        oos.models[i,] <- array(coef(fit[[1]])) * if(!is.numeric(Y)) 1 else scale
        oos.pred[which(foldid == i)] <- unlist(.cv_predict(fit, temp.X[which(foldid == i),]))
    }


    oos.pred <- if(!is.numeric(Y)) 1/(1+exp(-oos.pred)) else oos.pred*scale
    Y <- if(!is.numeric(Y)) Y else Y*scale
    
    #formatting iEN class objects
    cv.eval <- switch(eval,
                    "ROCAUC" =  vapply(1, FUN=.ROCAUC_eval, FUN.VALUE= numeric(1), preds=cbind(oos.pred,oos.pred), resp=Y),
                    "RSS" = vapply(1, FUN=.RSS_eval, FUN.VALUE= numeric(1), preds=cbind(oos.pred,oos.pred), resp=Y),
                    "RMSE" = vapply(1, FUN=.RMSE_eval, FUN.VALUE= numeric(1), preds=cbind(oos.pred,oos.pred), resp=Y),
                    "wilcox" = vapply(1, FUN=.wilcox_eval, FUN.VALUE= numeric(1), preds=cbind(oos.pred,oos.pred), resp=Y),
                    "spearman" = vapply(seq(numL), FUN=.spearman_eval, FUN.VALUE= numeric(1), preds=inner.cv.preds[,,1], resp=train.Y),
                    "pearson" = vapply(seq(numL), FUN=.pearson_eval, FUN.VALUE= numeric(1), preds=inner.cv.preds[,,1], resp=train.Y),
                    stop("Evaluation method not defined"))

    #for reporting purposes must transform RSS RMSE from inverse
    cv.eval <- switch(eval,
                    "ROCAUC" =  cv.eval,
                    "RSS" = 1/cv.eval,
                    "RMSE" = 1/cv.eval,
                    "wilcox" = cv.eval,
                    "spearman" = cv.eval,
                    "pearson" = cv.eval,
                    stop("Evaluation method not defined"))

    kNet <- new("iEN", betas=as(oos.models,"dgCMatrix"), nobs=nrow(X), nfeat=ncol(X), cv.preds=oos.pred, 
        cv.eval=cv.eval, dfs=rowSums(oos.models != 0), opt.params=max.params, call=match.call())

    return(kNet)
}

.kfold_prioritization_par_Function <- function(k, par.index, X, Y, nlambda, lambdas, foldid, eval, family, intercept, priors){
    folds <- unique(foldid)
    conditions <- list(is.null(lambdas) && nlambda > 0, !is.numeric(Y))
    numL <- ifelse(nlambda > 0, nlambda, length(lambdas))
    inner.cv.eval <- array(data=NA, dim=c(length(folds), numL, 2), dimnames=list(as.character(folds), NULL, c("evaluation","lambda")))

    for(i in folds){
        inner.folds <- foldid[which(foldid != i)]
        train.X <- X[which(foldid != i),]
        train.Y <- Y[which(foldid != i)]

        inner.cv.preds <- array(data=NA, dim=c(length(inner.folds), numL, 2), dimnames=list(rownames(train.X), NULL, NULL))

        for(j in unique(inner.folds)){
            scaled.train.X <- train.X %*% diag(exp(-(1-priors) * par.index[k,2]))

            if(conditions[[1]]){
                lambdas <- .gen_lambdas(scaled.train.X[which(inner.folds != j),], train.Y[which(inner.folds != j)], nlambda)
            }

            fits <- .iEN(scaled.train.X[which(inner.folds != j),], train.Y[which(inner.folds != j)], par.index[k,1], lambdas, family, intercept)
            temp <- .cv_predict(fits,scaled.train.X[which(inner.folds == j),])
            
            for(z in seq(length(temp))){
                inner.cv.preds[which(inner.folds == j), z, 1]  <- array(temp[[z]])
            }
            if(conditions[[2]]){
                inner.cv.preds[which(inner.folds == j), , 1] <- 1/(1+exp(-inner.cv.preds[which(inner.folds == j), , 1]))
            }

            inner.cv.preds[which(inner.folds == j), ,2] <- t(array(rep(lambdas, sum(inner.folds == j)), dim= c(numL,sum(inner.folds == j))))
        }

        inner.cv.eval[which(folds == i), ,1] <- switch(eval,
                    "ROCAUC" =  vapply(seq(numL), FUN=.ROCAUC_eval, FUN.VALUE= numeric(1), preds=inner.cv.preds[,,1], resp=train.Y),
                    "RSS" = vapply(seq(numL), FUN=.RSS_eval, FUN.VALUE= numeric(1), preds=inner.cv.preds[,,1], resp=train.Y),
                    "RMSE" = vapply(seq(numL), FUN=.RMSE_eval, FUN.VALUE= numeric(1), preds=inner.cv.preds[,,1], resp=train.Y),
                    "wilcox" = vapply(seq(numL), FUN=.wilcox_eval, FUN.VALUE= numeric(1), preds=inner.cv.preds[,,1], resp=train.Y),
                    "spearman" = vapply(seq(numL), FUN=.spearman_eval, FUN.VALUE= numeric(1), preds=inner.cv.preds[,,1], resp=train.Y),
                    "pearson" = vapply(seq(numL), FUN=.pearson_eval, FUN.VALUE= numeric(1), preds=inner.cv.preds[,,1], resp=train.Y),
                    stop("Evaluation method not defined"))
        inner.cv.eval[which(folds == i), ,2] <- colMeans(inner.cv.preds[,,2], na.rm=TRUE)
    }
    return(inner.cv.eval)
}

.gen_lambdas <- function(sX, sY, nlambda){
    if(is.numeric(sY)){
        lambda.max <- max(abs(colSums(sX*sY)),na.rm=TRUE)/nrow(sX)
    }else{
        temp.table <- prop.table(table(sY))
        lambda.max <- max(abs(colSums(sX*ifelse(sY == levels(sY)[1], -temp.table[2], temp.table[1]))),na.rm=TRUE)/nrow(sX)
    }
    return(round(exp(seq(log(lambda.max), log(lambda.max*0.00001), length.out=nlambda)), digits=10))
}

.iEN <- function(X, Y, alpha, lambdas, family=c("binomial","gaussian"), intercept){
    return(lapply(seq(length(lambdas)), FUN=function(i){
        return(glmnet(X, Y, alpha=alpha, lambda=lambdas[i], family=family, intercept=intercept, standardize=FALSE, standardize.response=FALSE))
    }))
}

.cv_predict <- function(fits, testX){
    return(lapply(seq(length(fits)), FUN=function(i){
        if(is.null(dim(testX))){
            return(predict(fits[[i]],t(testX)))
        }else{
            return(predict(fits[[i]],testX))
        }
    }))
}

.ROCAUC_eval <- function(i, preds, resp){
    pred.vec <- preds[,i]
    val <- roc(resp, pred.vec, direction = "<", quiet=TRUE)$auc

    if(sum(is.na(pred.vec)) > 2/5 * length(pred.vec)){
        return(-3)
    }else if(sum(pred.vec == 0) == length(pred.vec)){
        return(-2)
    }

    return(ifelse(val < 0.5, 0.5, val))
}

.RSS_eval <- function(i, preds, resp){
    pred.vec <- preds[,i]

    if(sum(is.na(pred.vec)) > 2/5 * length(pred.vec) ){
        return(-3)  
    }else if(sum(pred.vec == 0) == length(pred.vec)){
        return(-2)
    }else if(cor(resp, pred.vec, method = "spearman", use = "complete.obs") < 0){
        return(-1)
    }

    val <- sum((resp - pred.vec)^2)
    return(1/val)
}

.RMSE_eval <- function(i, preds, resp){
    pred.vec <- preds[,i]

    if(sum(is.na(pred.vec)) > 2/5 * length(pred.vec)){
        return(-3)
    }else if(sum(pred.vec == 0) == length(pred.vec)){
        return(-2)
    }else if(cor(resp, pred.vec, method = "spearman", use = "complete.obs") < 0 ){
        return(-1)
    }
    return(1/rmse(resp, pred.vec))
}

.wilcox_eval <- function(i, preds, resp){
    pred.vec <- preds[,i]
    pv <- wilcox.test(pred.vec~resp)$p.value

    if(sum(is.na(pred.vec)) > 2/5 * length(pred.vec) ){
        return(-3)
    }else if(sum(pred.vec == 0) == length(pred.vec)){
        return(-2)
    }

    return(-log10(pv))
}

.pearson_eval <- function(i, preds, resp){
    pred.vec <- preds[,i]
    pv <- cor.test(resp, pred.vec, method = "pearson", use = "complete.obs")$p.value

    if(sum(is.na(pred.vec)) > 2/5 * length(pred.vec)){
        return(-3)
    }else if(sum(pred.vec == 0) == length(pred.vec)){
        return(-2)
    }else if(cor(resp, pred.vec, method = "spearman", use = "complete.obs") < 0 ){
        return(-1)
    }
    return(-log10(pv))
}



.spearman_eval <- function(i, preds, resp){
    pred.vec <- preds[,i]
    pv <- cor.test(resp, pred.vec, method = "spearman", use = "complete.obs")$p.value

    if(sum(is.na(pred.vec)) > 2/5 * length(pred.vec)){
        return(-3)
    }else if(sum(pred.vec == 0) == length(pred.vec)){
        return(-2)
    }else if(cor(resp, pred.vec, method = "spearman", use = "complete.obs") < 0 ){
        return(-1)
    }
    return(-log10(pv))
}
