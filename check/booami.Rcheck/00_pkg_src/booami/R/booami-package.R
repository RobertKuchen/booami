#' Boosting with Multiple Imputation (booami)
#'
#' **booami** provides component-wise gradient boosting tailored for analysis
#' with multiply imputed datasets. Its core contribution is **MIBoost**, a novel
#' algorithm that couples base-learner selection across imputed datasets by
#' minimizing an aggregated loss at each iteration, yielding a single, unified
#' regularization path and improved model stability. For comparison,
#' \pkg{booami} also includes per-dataset boosting with post-hoc pooling
#' (estimate averaging or selection-frequency thresholding).
#'
#' ## What is MIBoost?
#' In each boosting iteration, candidate base-learners are fit separately within
#' each imputed dataset, but selection is made **jointly** via the aggregated
#' loss across datasets. The selected base-learner is then updated in every
#' imputed dataset, and fitted contributions are averaged to form a single
#' combined predictor. This enforces uniform variable selection while preserving
#' dataset-specific gradients and updates.
#'
#' ## Cross-validation without leakage
#' \pkg{booami} implements a leakage-avoiding CV protocol:
#' data are first split into training and validation subsets; training data are
#' multiply imputed; validation data are imputed using the **training** imputation
#' models; and centering uses training means. Errors are averaged across
#' imputations and folds to select the optimal number of boosting iterations
#' (\code{mstop}). Use \code{\link{cv_boost_raw}} for raw data with missing values
#' (imputation inside CV), or \code{\link{cv_boost_imputed}} when imputed datasets
#' are already prepared.
#'
#' ## Key features
#' - **MIBoost (uniform selection):** Joint base-learner selection via aggregated
#'   loss across imputed datasets; averaged fitted functions yield a single model.
#' - **Per-dataset boosting (with pooling):** Independent boosting in each
#'   imputed dataset, with pooling by estimate averaging or by
#'   selection-frequency thresholding.
#' - **Flexible losses and learners:** Supports Gaussian and logistic losses with
#'   component-wise base-learners; extensible to other learners.
#' - **Leakage-safe CV:** Training/validation split → train-only imputation →
#'   training-mean centering → error aggregation across imputations.
#'
#' ## Main functions
#' - \code{\link{impu_boost}} — Core routine implementing **MIBoost** as well as
#'   per-dataset boosting with pooling.
#' - \code{\link{cv_boost_raw}} — Leakage-safe k-fold CV starting from a single
#'   dataset with missing values (imputation performed inside each fold).
#' - \code{\link{cv_boost_imputed}} — CV when imputed datasets (and splits) are
#'   already available.
#'
#' ## Typical workflow
#' 1. **Raw data with missingness:** use \code{cv_boost_raw()} to impute within
#'    folds, select \code{mstop}, and fit the final model.
#' 2. **Already imputed datasets:** use \code{cv_boost_imputed()} to select
#'    \code{mstop} and fit.
#' 3. **Direct control:** call \code{impu_boost()} when you want to run
#'    MIBoost (or per-dataset boosting) directly, optionally followed by pooling.
#'
#' ## Mathematical sketch
#' At boosting iteration \eqn{t}, for each candidate base-learner \eqn{r} and
#' each imputed dataset \eqn{m = 1,\dots,M}, let
#' \eqn{RSS_r^{(m)[t]}} denote the residual sum of squares.
#' The aggregated loss is
#' \deqn{L_r^{[t]} = \sum_{m=1}^M RSS_r^{(m)[t]}.}
#' The base-learner \eqn{r^*} with minimal aggregated loss is selected jointly,
#' updated in all imputed datasets, and the fitted contributions are averaged to
#' form the combined predictor. After \eqn{t_{\mathrm{stop}}} iterations, this
#' yields a single final model.
#'
#' ## Citation
#' For details, see: Kuchen, R. (2025). *MIBoost: A Gradient Boosting Algorithm
#' for Variable Selection After Multiple Imputation*. arXiv:2507.21807.
#' \doi{10.48550/arXiv.2507.21807} \url{https://arxiv.org/abs/2507.21807}.
#'
#' ## See also
#' - \pkg{mboost}: General framework for component-wise gradient boosting in R.
#' - \pkg{miselect}: Implements MI-extensions of LASSO and elastic nets for
#'   variable selection after multiple imputation.
#' - \pkg{mice}: Standard tool for multiple imputation of missing data.
#'
#' @keywords internal
"_PACKAGE"
