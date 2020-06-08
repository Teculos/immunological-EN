test.iEN <- function(){
    library(RUnit)
    library(iEN)

    alphaGrid <- seq(0,1, length.out=3)
    phiGrid <- exp(seq(log(1), log(100), length.out=3))
    nlambda <- 3
    ncores <- 7
    eval <- "RSS" 
    family <- "gaussian"
    intercept <- TRUE
    standardize <- TRUE
    center <- TRUE

    data(test_data)

    #training models
    ien.ans=cv_iEN(X, Y, foldid, alphaGrid, phiGrid, nlambda, NULL, priors, ncores, eval, family, intercept, standardize, center)
    en.ans=cv_iEN(X, Y, foldid, alphaGrid, phiGrid=c(0,0), nlambda, NULL, priors, ncores, eval, family, intercept, standardize, center)

    checkEquals(as.vector(ien.ans@nobs),length(Y))
    checkEquals(as.vector(ien.ans@nfeat),ncol(X))
    checkEquals(dim(ien.ans@opt.params), c(length(unique(foldid)), 3),tolerance=10^-6)
    checkTrue(-log10(cor.test(ien.ans@cv.preds,Y)$p.value) > -log10(cor.test(en.ans@cv.preds,Y)$p.value))
}
