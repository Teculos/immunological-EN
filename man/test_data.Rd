\name{test_data}
\alias{test_data}
\alias{X}
\alias{Y}
\alias{priors}
\alias{foldid}
\docType{data}
\title{test_data}
\description{
A dataset containing CyTOF data relating to a study on the immunological progression of pregnancy. Includes immune features X and corresponding gestational age Y, as well as corresponding biological priors and fold identification per individual to maintain patient separation during cross-validation.
}
\usage{data(test_data)}
\details{This dataset consists of a single data matrix and three vectors
\code{X} CyTOF data 
\code{Y} Response vector of gestational age at time of sampling
\code{priors} Priors indicating prioritization of immune features
\code{foldid} identification of individuals for Cross-Validation
}

\author{
Anthony Culos \email{<tculos@stanford.edu>} and Nima Aghaeepour \email{<naghaeep@gmail.com>}.
}

\references{https://nalab.stanford.edu/}

\examples{
data(test_data)
}
