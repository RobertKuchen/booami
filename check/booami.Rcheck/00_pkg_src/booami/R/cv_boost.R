#' Cross-validated boosting on already-imputed data
#'
#' Performs k-fold cross-validation for \code{\link{impu_boost}} to determine
#' the optimal value of \code{mstop} before fitting the final model on the
#' full dataset. This function should only be used when data have already
#' been imputed. In most cases, it is preferable to provide unimputed data
#' and use \code{\link{cv_boost_raw}} instead.
#'
#' To avoid data leakage, each CV fold should first be split into training and
#' validation subsets, after which imputation is performed. For the final model,
#' all data should be imputed independently.
#'
#' The recommended workflow is illustrated in the examples.
#'
#' @references
#' Kuchen, R. (2025). \emph{MIBoost: A Gradient Boosting Algorithm for Variable
#' Selection After Multiple Imputation}. arXiv:2507.21807.
#' \doi{10.48550/arXiv.2507.21807} \url{https://arxiv.org/abs/2507.21807}.
#'
#' @param X_train_list A list of length \eqn{k}. Element \code{i} is itself a
#'   list of length \eqn{M} containing the \eqn{n_{train} \times p} numeric
#'   design matrices for each imputed dataset in CV fold \code{i}.
#' @param y_train_list A list of length \eqn{k}. Element \code{i} is a list of
#'   length \eqn{M}, where each element is a length-\eqn{n_{train}} numeric
#'   response vector aligned with \code{X_train_list[[i]][[m]]}.
#' @param X_val_list A list of length \eqn{k}. Element \code{i} is a list of
#'   length \eqn{M} containing the \eqn{n_{val} \times p} numeric validation
#'   matrices matched to the corresponding imputed training dataset
#'   in fold \code{i}.
#' @param y_val_list A list of length \eqn{k}. Element \code{i} is a list of
#'   length \eqn{M}, where each element is a length-\eqn{n_{val}} continuous
#'   response vector aligned with \code{X_val_list[[i]][[m]]}.
#' @param X_full A list of length \eqn{M} with the \eqn{n \times p} numeric
#'   full-data design matrices (one per imputed dataset) used to fit
#'   the final model.
#' @param y_full A list of length \eqn{M}, where each element is a
#'   length-\eqn{n} continuous response vector corresponding to the
#'   imputed dataset in \code{X_full}.
#' @param ny Learning rate. Defaults to \code{0.1}.
#' @param mstop Maximum number of boosting iterations to evaluate during
#'   cross-validation. The selected \code{mstop} is the value that minimizes
#'   the mean CV error over \code{1:mstop}. Default is \code{250}.
#' @param type Type of loss function. One of:
#'   \code{"gaussian"} (mean squared error) for continuous responses,
#'   or \code{"logistic"} (binomial deviance) for binary responses.
#' @param MIBoost Logical. If \code{TRUE}, applies the MIBoost algorithm,
#'   which enforces uniform variable selection across all imputed datasets. If
#'   \code{FALSE}, variables are selected independently within each imputed
#'   dataset, and pooling is governed by \code{pool_threshold}.
#' @param pool Logical. If \code{TRUE}, models across the \eqn{M} imputed
#'   datasets are aggregated into a single final model. If \code{FALSE},
#'   \eqn{M} separate models are returned.
#' @param pool_threshold Only used when \code{MIBoost = FALSE} and \code{pool = TRUE}.
#'   Controls the pooling rule when aggregating the \eqn{M} models obtained from
#'   the imputed datasets into a single final model. A candidate variable is
#'   included only if it is selected in at least \code{pool_threshold} (a value
#'   in (0, 1]) proportion of the imputed datasets; coefficients of all other
#'   variables are set to zero. A value of \code{0} corresponds to
#'   estimate-averaging, while values \code{> 0} correspond to
#'   selection-frequency thresholding.
#' @param show_progress Logical; print fold-level progress and summary timings.
#'   Default \code{TRUE}.
#' @param center One of \code{c("auto", "off", "force")}. Controls
#'   centering of \code{X} within each imputed dataset.
#'   With \code{"auto"} (recommended), centering is applied only if the training
#'   matrix is not already centered. With \code{"force"}, centering is always
#'   applied. With \code{"off"}, centering is skipped. If \code{X_val_list} is
#'   provided, validation sets are centered using the means from the
#'   corresponding training set.
#'
#' @details
#' Centering affects only \code{X}; \code{y} is left unchanged. For
#' \code{type = "logistic"}, responses are treated as numeric \code{0/1}
#' via the logistic link. Validation loss is averaged over
#' imputations and then over folds.
#'
#' @return A list with:
#' \itemize{
#'   \item \code{CV_error}: numeric vector of length \code{mstop} with the mean
#'         cross-validated loss across folds (and imputations).
#'   \item \code{best_mstop}: integer index of the minimizing entry in
#'         \code{CV_error}.
#'   \item \code{final_model}: numeric vector of length \code{1 + p} containing
#'         the intercept followed by \eqn{p} coefficients of the final pooled
#'         model fitted at \code{best_mstop} on \code{X_full}/\code{y_full}.
#' }
#'
#' @seealso \code{\link{impu_boost}}, \code{\link{cv_boost_raw}}
#'
#' @examplesIf requireNamespace("mice", quietly = TRUE) && requireNamespace("miceadds", quietly = TRUE)
#' \donttest{
#'   set.seed(123)
#'   utils::data(booami_sim)
#'   k <- 2; M <- 2
#'   n <- nrow(booami_sim); p <- ncol(booami_sim) - 1
#'   folds <- sample(rep(seq_len(k), length.out = n))
#'
#'   X_train_list <- vector("list", k)
#'   y_train_list <- vector("list", k)
#'   X_val_list   <- vector("list", k)
#'   y_val_list   <- vector("list", k)
#'
#'   for (cv in seq_len(k)) {
#'     tr <- folds != cv
#'     va <- !tr
#'     dat_tr <- booami_sim[tr, , drop = FALSE]
#'     dat_va <- booami_sim[va, , drop = FALSE]
#'     pm_tr  <- mice::quickpred(dat_tr, method = "spearman", mincor = 0.30, minpuc = 0.60)
#'     imp_tr <- mice::mice(dat_tr, m = M, predictorMatrix = pm_tr, maxit = 1, printFlag = FALSE)
#'     imp_va <- mice::mice.mids(imp_tr, newdata = dat_va, maxit = 1, printFlag = FALSE)
#'     X_train_list[[cv]] <- vector("list", M)
#'     y_train_list[[cv]] <- vector("list", M)
#'
#'     X_val_list[[cv]]   <- vector("list", M)
#'     y_val_list[[cv]]   <- vector("list", M)
#'     for (m in seq_len(M)) {
#'       tr_m <- mice::complete(imp_tr, m)
#'       va_m <- mice::complete(imp_va, m)
#'       X_train_list[[cv]][[m]] <- data.matrix(tr_m[, 1:p, drop = FALSE])
#'       y_train_list[[cv]][[m]] <- tr_m$y
#'       X_val_list[[cv]][[m]]   <- data.matrix(va_m[, 1:p, drop = FALSE])
#'       y_val_list[[cv]][[m]]   <- va_m$y
#'     }
#'   }
#'
#'   pm_full  <- mice::quickpred(booami_sim, method = "spearman", mincor = 0.30, minpuc = 0.60)
#'   imp_full <- mice::mice(booami_sim, m = M, predictorMatrix = pm_full, maxit = 1, printFlag = FALSE)
#'   X_full <- lapply(seq_len(M),
#'   function(m) data.matrix(
#'   mice::complete(imp_full, m)[, 1:p, drop = FALSE]))
#'   y_full <- lapply(seq_len(M), function(m) mice::complete(imp_full, m)$y)
#'
#'   res <- cv_boost_imputed(
#'     X_train_list, y_train_list,
#'     X_val_list,   y_val_list,
#'     X_full,       y_full,
#'     ny = 0.1, mstop = 50, type = "gaussian",
#'     MIBoost = TRUE, pool = TRUE, center = "auto",
#'     show_progress = FALSE
#'   )
#' \dontshow{invisible(utils::capture.output(str(res)))}
#' }
#'
#' \dontrun{
#'   set.seed(2025)
#'   utils::data(booami_sim)
#'   k <- 5; M <- 10
#'   n <- nrow(booami_sim); p <- ncol(booami_sim) - 1
#'   folds <- sample(rep(seq_len(k), length.out = n))
#'
#'   X_train_list <- vector("list", k)
#'   y_train_list <- vector("list", k)
#'   X_val_list   <- vector("list", k)
#'   y_val_list   <- vector("list", k)
#'   for (cv in seq_len(k)) {
#'     tr <- folds != cv; va <- !tr
#'     dat_tr <- booami_sim[tr, , drop = FALSE]
#'     dat_va <- booami_sim[va, , drop = FALSE]
#'     pm_tr  <- mice::quickpred(dat_tr, method = "spearman", mincor = 0.20, minpuc = 0.40)
#'     imp_tr <- mice::mice(dat_tr, m = M, predictorMatrix = pm_tr, maxit = 5, printFlag = TRUE)
#'     imp_va <- mice::mice.mids(imp_tr, newdata = dat_va, maxit = 1, printFlag = FALSE)
#'     X_train_list[[cv]] <- vector("list", M)
#'     y_train_list[[cv]] <- vector("list", M)
#'     X_val_list[[cv]]   <- vector("list", M)
#'     y_val_list[[cv]]   <- vector("list", M)
#'     for (m in seq_len(M)) {
#'       tr_m <- mice::complete(imp_tr, m); va_m <- mice::complete(imp_va, m)
#'       X_train_list[[cv]][[m]] <- data.matrix(tr_m[, 1:p, drop = FALSE])
#'       y_train_list[[cv]][[m]] <- tr_m$y
#'       X_val_list[[cv]][[m]]   <- data.matrix(va_m[, 1:p, drop = FALSE])
#'       y_val_list[[cv]][[m]]   <- va_m$y
#'     }
#'   }
#'   pm_full  <- mice::quickpred(booami_sim, method = "spearman", mincor = 0.20, minpuc = 0.40)
#'   imp_full <- mice::mice(booami_sim, m = M, predictorMatrix = pm_full, maxit = 5, printFlag = TRUE)
#'   X_full <- lapply(seq_len(M),
#'   function(m) data.matrix(mice::complete(imp_full, m)[, 1:p, drop = FALSE]))
#'   y_full <- lapply(seq_len(M),
#'   function(m) mice::complete(imp_full, m)$y)
#'
#'   res_heavy <- cv_boost_imputed(
#'     X_train_list, y_train_list,
#'     X_val_list,   y_val_list,
#'     X_full,       y_full,
#'     ny = 0.1, mstop = 250, type = "gaussian",
#'     MIBoost = TRUE, pool = TRUE, center = "auto",
#'     show_progress = TRUE
#'   )
#'   str(res_heavy)
#' }
#' @export
#' @export
cv_boost_imputed <- function(
    X_train_list, y_train_list,
    X_val_list,   y_val_list,
    X_full, y_full,
    ny = 0.1, mstop = 250, type = c("gaussian","logistic"),
    MIBoost = TRUE, pool = TRUE, pool_threshold = 0,
    show_progress = TRUE,
    center = c("auto","off","force")
) {
  # match args (keep same order/choices as core)
  type   <- match.arg(type)
  center <- match.arg(center)

  # ---- light validation (fold-level) ----
  K <- length(y_train_list)
  stopifnot(
    length(X_train_list) == K,
    length(X_val_list)   == K,
    length(y_val_list)   == K,
    K >= 2L
  )
  if (!is.numeric(ny) || length(ny) != 1L || !is.finite(ny) || ny <= 0) {
    stop("`ny` must be a single positive numeric.")
  }
  if (!is.numeric(mstop) || length(mstop) != 1L || !is.finite(mstop) || mstop < 1) {
    stop("`mstop` must be a single positive integer.")
  }
  mstop <- as.integer(mstop)

  # ensure each fold has same number of imputations M and basic shapes look sane
  Ms <- integer(K)
  for (cv in seq_len(K)) {
    Ms[cv] <- length(X_train_list[[cv]])
    if (length(y_train_list[[cv]]) != Ms[cv] ||
        length(X_val_list[[cv]])   != Ms[cv] ||
        length(y_val_list[[cv]])   != Ms[cv]) {
      stop(sprintf("Fold %d: inconsistent number of imputations across X/y train/val.", cv))
    }
  }
  if (length(unique(Ms)) != 1L) stop("All folds must have the same number of imputations M.")
  M <- Ms[1L]

  # full data lists should also have M imputations
  if (length(X_full) != M || length(y_full) != M) {
    stop("`X_full` and `y_full` must each be lists of length M (same M as the folds).")
  }

  # basic logistic target check (silent coercion not performed)
  if (type == "logistic") {
    ok01 <- function(v) all(is.finite(v)) && all(v %in% c(0,1))
    if (!ok01(y_train_list[[1]][[1]]) || !ok01(y_val_list[[1]][[1]]) || !ok01(y_full[[1]])) {
      stop("For type='logistic', outcomes must be 0/1 (no NAs, no other values).")
    }
  }

  # ---- delegate to the core ----
  res <- .cv_boost_core(
    X_train_list = X_train_list, y_train_list = y_train_list,
    X_val_list   = X_val_list,   y_val_list   = y_val_list,
    X = X_full, y = y_full,
    ny = ny, mstop = mstop, type = type,
    MIBoost = MIBoost, pool = pool, pool_threshold = pool_threshold,
    show_progress = show_progress,
    center = center
  )


  # ---- friendly metadata & names for prediction -------------------
  # 1) Annotate family type for booami_predict() inference
  if (is.null(res$type)) res$type <- type

  # 2) Bubble up center_means if the core exposed them (preferred)
  #    a) directly on the returned object
  #    b) or nested in a `final_fit` list (if the core returns the impu_boost fit)
  if (is.null(res$center_means)) {
    if (!is.null(res$final_fit) && !is.null(res$final_fit$center_means)) {
      res$center_means <- res$final_fit$center_means
    }
  }
  # (No attempt here to "recompute" means from X_full - we only propagate what the core actually used.)

  # 3) Name final_model coefficients for name-based alignment in prediction
  if (!is.null(res$final_model)) {
    pred_names <- colnames(X_full[[1L]])
    if (!is.null(pred_names) &&
        length(res$final_model) == (length(pred_names) + 1L) &&
        is.null(names(res$final_model))) {
      names(res$final_model) <- c("(Intercept)", pred_names)
    }
  }

  class(res) <- c("booami_cv_imputed", "booami_cv", class(res))

  res
}



#' Cross-Validated Component-Wise Gradient Boosting with
#' Multiple Imputation Performed Inside Each Fold
#'
#' Performs k-fold cross-validation for \code{\link{impu_boost}} on data with
#' missing values. Within each fold, multiple imputation, centering, model
#' fitting, and validation are performed in a leakage-avoiding manner to select
#' the optimal number of boosting iterations (\code{mstop}). The final model is
#' then fitted on multiple imputations of the full dataset at the selected
#' stopping iteration.
#'
#' @details
#' Within each CV fold, the data are first split into a training subset and a
#' validation subset. The training subset is multiply imputed \eqn{M} times
#' using \pkg{mice}, producing \eqn{M} imputed training datasets. Covariates
#' in each training dataset are centered. The corresponding validation subset
#' is then imputed \eqn{M} times using the imputation models learned from the
#' training imputations, ensuring consistency between training and validation.
#' These validation datasets are centered using the variable means from their
#' associated training datasets.
#'
#' \code{\link{impu_boost}} is run on the imputed training datasets for up to
#' \code{mstop} boosting iterations. At each iteration, prediction errors are
#' computed on the corresponding validation datasets and averaged across
#' imputations. This yields an aggregated error curve per fold, which is then
#' averaged across folds. The optimal stopping iteration is chosen as the
#' \code{mstop} value minimizing the mean CV error.
#'
#' Finally, the full dataset is multiply imputed \eqn{M} times and centered
#' independently within each imputed dataset. \code{\link{impu_boost}} is
#' applied to these datasets for the selected number of boosting iterations to
#' obtain the final model.
#'
#' \strong{Imputation control.} All key \pkg{mice} settings can be passed via
#' \code{impute_args} (a named list forwarded to \code{mice::mice()}) and/or
#' \code{impute_method} (a named character vector of per-variable methods).
#' Internally, the function builds a full default method vector from the actual
#' data given to \code{mice()}, then \emph{merges} any user-supplied entries
#' by name. \emph{The names in \code{impute_method} must exactly match the
#' column names in \code{data.frame(y = y, X)} (i.e., the data passed
#' to \code{mice()}).} Partial vectors are allowed; variables not listed fall
#' back to defaults; unknown names are ignored with a warning. The function sets
#' and may override \code{data}, \code{method} (after merging overrides),
#' \code{predictorMatrix}, and \code{ignore} (to enforce train-only learning).
#' Predictor matrices can be built with \code{mice::quickpred()} (see
#' \code{use_quickpred}, \code{quickpred_args}) or with
#' \code{mice::make.predictorMatrix()}.
#'
#' @references
#' Kuchen, R. (2025). \emph{MIBoost: A Gradient Boosting Algorithm for Variable
#' Selection After Multiple Imputation}. arXiv:2507.21807.
#' \doi{10.48550/arXiv.2507.21807} \url{https://arxiv.org/abs/2507.21807}.
#'
#' @param X A data.frame or matrix of predictors of size \eqn{n \times p}
#'   containing missing values. Column names are preserved. If no missing values
#'   are present in \code{X} or \code{y}, use
#'   \code{\link{cv_boost_imputed}} instead.
#' @param y A vector of length \eqn{n} with the outcome (numeric for
#'   \code{type = "gaussian"}; numeric \code{0/1} or a 2-level factor for
#'   \code{type = "logistic"}). Must align with \code{X} rows.
#' @param k Number of cross-validation folds. Default is \code{5}.
#' @param ny Learning rate. Defaults to \code{0.1}.
#' @param mstop Maximum number of boosting iterations to evaluate during
#'   cross-validation. The selected \code{mstop} is the value minimizing
#'   the mean CV error over \code{1:mstop}. Default is \code{250}.
#' @param type Type of loss function. One of:
#'   \code{"gaussian"} (mean squared error) for continuous responses,
#'   or \code{"logistic"} (binomial deviance) for binary responses.
#' @param MIBoost Logical. If \code{TRUE}, applies the MIBoost algorithm,
#'   which enforces uniform variable selection across all imputed datasets. If
#'   \code{FALSE}, variables are selected independently within each imputed
#'   dataset, and pooling is governed by \code{pool_threshold}.
#' @param pool Logical. If \code{TRUE}, models across the \eqn{M} imputed
#'   datasets are aggregated into a single final model. If \code{FALSE},
#'   \eqn{M} separate models are returned.
#' @param pool_threshold Only used when \code{MIBoost = FALSE} and
#'   \code{pool = TRUE}. Controls the pooling rule when aggregating the
#'   \eqn{M} models obtained from the imputed datasets into a single final model.
#'   A candidate variable is included only if it is selected in at least
#'   \code{pool_threshold} (a value in (0, 1) proportion of the imputed
#'   datasets; coefficients of all other variables are set to zero. A value of
#'   \code{0} corresponds to estimate-averaging, while values \code{> 0}
#'   correspond to selection-frequency thresholding.
#' @param impute_args A named list of arguments forwarded to \code{mice::mice()}
#'   both inside CV and on the full dataset (e.g., \code{m}, \code{maxit},
#'   \code{seed}, \code{printFlag}, etc.). Internally, \code{data},
#'   \code{predictorMatrix}, and \code{ignore} are set by the function and will
#'   override any values supplied here. If \code{m} is missing, a default of
#'   \code{10} is used.
#' @param impute_method Optional \emph{named} character vector passed to
#'   \code{mice::mice(method = ...)} to control per-variable methods
#'   (e.g., \code{"pmm"}, \code{"logreg"}). This may be a \emph{partial}
#'   vector: entries are merged by name into a full default method vector
#'   derived from the data; unmatched names are ignored with a warning.
#'   If \code{NULL} (default), numeric columns use \code{"pmm"}; for
#'   \code{type = "logistic"}, the outcome uses \code{"logreg"} (coerced to a
#'   2-level factor if needed).
#' @param use_quickpred Logical. If \code{TRUE} (default), build the
#'   \code{predictorMatrix} via \code{mice::quickpred()} on the training data
#'   within each fold; otherwise use \code{mice::make.predictorMatrix()}.
#' @param quickpred_args A named list of arguments forwarded to
#'   \code{mice::quickpred()} (e.g., \code{mincor}, \code{minpuc},
#'   \code{method}, \code{include}, \code{exclude}). Ignored when
#'   \code{use_quickpred = FALSE}.
#' @param seed Base random seed for fold assignment. If \code{impute_args$seed}
#'   is \emph{not} supplied, this value also seeds imputation; otherwise the
#'   user-specified \code{impute_args$seed} is respected and deterministically
#'   offset per fold. RNG state is restored on exit. Default \code{123}.
#' @param show_progress Logical. If \code{TRUE} (default), print progress for
#'   the imputation and boosting phases, plus a summary at completion.
#' @param return_full_imputations Logical. If \code{TRUE}, attach the list of
#'   full-data imputations used for the final fit as
#'   \code{$full_imputations = list(X = <list length m>, y = <list length m>)}.
#'   Default is \code{FALSE}.
#' @param center One of \code{c("auto", "off", "force")}. Controls
#'   centering of \code{X} within each imputed dataset.
#'   With \code{"auto"} (recommended), centering is applied only if the training
#'   matrix is not already centered. With \code{"force"}, centering is always
#'   applied. With \code{"off"}, centering is skipped. Validation sets are
#'   always centered using the means from the corresponding training set.
#'
#' @return A list with:
#' \itemize{
#'   \item \code{CV_error}: numeric vector (length \code{mstop}) of mean CV loss.
#'   \item \code{best_mstop}: integer index minimizing \code{CV_error}.
#'   \item \code{final_model}: numeric vector of length \code{1 + p} with the
#'         intercept and pooled coefficients of the final fit on full-data
#'         imputations at \code{best_mstop}.
#'   \item \code{full_imputations}: (optional) when \code{return_full_imputations=TRUE},
#'         a list \code{list(X = <list length m>, y = <list length m>)} containing
#'         the full-data imputations used for the final model.
#'   \item \code{folds}: integer vector of length \eqn{n} giving the CV fold id
#'         for each observation (1..k).
#' }
#'
#' @seealso \code{\link{impu_boost}}, \code{\link{cv_boost_imputed}}, \pkg{mice}
#'
#' @examplesIf requireNamespace("mice", quietly = TRUE) && requireNamespace("miceadds", quietly = TRUE)
#' \donttest{
#'   utils::data(booami_sim)
#'   X <- booami_sim[, 1:25]
#'   y <- booami_sim[, 26]
#'
#'   res <- cv_boost_raw(
#'     X = X, y = y,
#'     k = 2, seed = 123,
#'     impute_args    = list(m = 2, maxit = 1, printFlag = FALSE, seed = 1),
#'     quickpred_args = list(mincor = 0.30, minpuc = 0.60),
#'     mstop = 50,
#'     show_progress = FALSE
#'   )
#'   \dontshow{invisible(utils::capture.output(str(res)))}
#'
#'   # Partial custom imputation method override
#'   meth <- c(y = "pmm", X1 = "pmm")
#'   res2 <- cv_boost_raw(
#'     X = X, y = y,
#'     k = 2, seed = 123,
#'     impute_args    = list(m = 2, maxit = 1, printFlag = FALSE, seed = 456),
#'     quickpred_args = list(mincor = 0.30, minpuc = 0.60),
#'     mstop = 50,
#'     impute_method  = meth,
#'     show_progress = FALSE
#'   )
#'   \dontshow{invisible(utils::capture.output(str(res2)))}
#' }
#'
#' @export
cv_boost_raw <- function(
    X, y,
    k = 5, ny = 0.1, mstop = 250, type = c("gaussian","logistic"),
    MIBoost = TRUE, pool = TRUE, pool_threshold = 0,
    impute_args = list(m = 10, maxit = 5, printFlag = FALSE),
    impute_method = NULL,
    use_quickpred = TRUE,
    quickpred_args = list(mincor = 0.1, minpuc = 0.5, method = NULL, include = NULL, exclude = NULL),
    seed = 123,
    show_progress = TRUE,
    return_full_imputations = FALSE,
    center = "auto"
) {
  type   <- match.arg(type)
  center <- match.arg(center, c("auto","off","force"))

  if (!is.data.frame(X) && !is.matrix(X))
    stop("X must be a data.frame or matrix.")
  X <- as.data.frame(X, check.names = FALSE)

  if (length(y) != nrow(X))
    stop("Length of y must match nrow(X).")

  if (!anyNA(X) && !anyNA(y))
    stop("Data have no missing values. Use cv_boost_imputed().")

  if (!("ignore" %in% names(formals(mice::mice))))
    stop("Update 'mice': need 'ignore' in mice().")

  # ---- helper: scoped RNG seed (restore on exit) --------------------
  .local_with_seed <- function(s) {
    if (is.null(s)) return(invisible(NULL))
    had <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    if (had) old <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    set.seed(s)
    on.exit({
      if (had) assign(".Random.seed", old, envir = .GlobalEnv)
    }, add = TRUE)
    invisible(NULL)
  }

  # ---- folds (scoped RNG) -------------------------------------------
  n <- nrow(X)
  .local_with_seed(seed)
  folds_id <- sample(rep(seq_len(k), length.out = n))

  # base seed for mice: respect user's impute_args$seed if present
  seed_base_mice <- if (!is.null(impute_args$seed)) impute_args$seed else seed

  get_m <- function() if (!is.null(impute_args$m)) impute_args$m else 10L

  build_methods <- function(dat) {
    meth <- mice::make.method(dat)
    is_num <- vapply(dat, is.numeric, logical(1L))
    meth[is_num] <- "pmm"
    if ("y" %in% names(dat)) {
      meth["y"] <- if (type == "gaussian") "pmm" else "logreg"
    }
    if (!is.null(impute_method)) {
      nm <- intersect(names(impute_method), names(meth))
      if (length(nm) == 0L) {
        warning("None of the names in 'impute_method' matched the data columns; overrides ignored.")
      } else {
        meth[nm] <- impute_method[nm]
      }
    }
    meth
  }

  as01 <- function(y) {
    if (is.factor(y) || is.character(y)) return(as.numeric(as.character(y)))
    as.numeric(y)
  }

  m <- get_m()
  if (isTRUE(show_progress)) cat(sprintf("MI-in-CV: %d folds  x  %d imputations...\n", k, m))

  X_train_list <- vector("list", k); y_train_list <- vector("list", k)
  X_val_list   <- vector("list", k); y_val_list   <- vector("list", k)

  for (i in seq_len(k)) {
    tr_idx <- folds_id != i; va_idx <- !tr_idx
    Xtr <- X[tr_idx, , drop = FALSE]; ytr <- y[tr_idx]
    Xva <- X[va_idx, , drop = FALSE]; yva <- y[va_idx]

    dat_tr  <- data.frame(y = ytr, Xtr, check.names = FALSE)
    dat_va  <- data.frame(y = yva, Xva, check.names = FALSE)
    dat_all <- rbind(dat_tr, dat_va)
    ntr <- nrow(dat_tr)

    if (isTRUE(use_quickpred)) {
      qp <- quickpred_args; qp <- qp[!vapply(qp, is.null, logical(1))]
      pm <- do.call(mice::quickpred, c(list(data = dat_tr), qp))
    } else {
      pm <- mice::make.predictorMatrix(dat_tr)
    }

    if (type == "logistic" && is.numeric(dat_all$y) &&
        all(stats::na.omit(dat_all$y) %in% c(0,1))) {
      dat_all$y <- factor(dat_all$y, levels = c(0,1))
    }

    meth <- build_methods(dat_all)

    ignore_vec <- c(rep(FALSE, ntr), rep(TRUE, nrow(dat_va)))
    args <- utils::modifyList(impute_args, list(
      data   = dat_all,
      method = meth,
      predictorMatrix = pm,
      ignore = ignore_vec
    ), keep.null = TRUE)
    if (is.null(args$m)) args$m <- m
    # seed: use user's base if provided, else ours; offset per fold
    if (is.null(args$seed)) {
      args$seed <- seed_base_mice + i
    } else {
      args$seed <- args$seed + i
    }

    if (isTRUE(show_progress)) cat(sprintf("Fold %d: mice (m=%d)...\n", i, args$m))
    mids <- do.call(mice::mice, args)

    X_train_list[[i]] <- vector("list", args$m)
    y_train_list[[i]] <- vector("list", args$m)
    X_val_list[[i]]   <- vector("list", args$m)
    y_val_list[[i]]   <- vector("list", args$m)

    for (j in seq_len(args$m)) {
      comp <- mice::complete(mids, action = j)
      comp_tr <- comp[seq_len(ntr), , drop = FALSE]
      comp_va <- comp[(ntr + 1):nrow(comp), , drop = FALSE]
      X_train_list[[i]][[j]] <- data.matrix(comp_tr[, setdiff(names(comp_tr), "y"), drop = FALSE])
      X_val_list[[i]][[j]]   <- data.matrix(comp_va[, setdiff(names(comp_va), "y"), drop = FALSE])
      y_train_list[[i]][[j]] <- if (type == "logistic") as01(comp_tr[["y"]]) else comp_tr[["y"]]
      y_val_list[[i]][[j]]   <- if (type == "logistic") as01(comp_va[["y"]]) else comp_va[["y"]]
    }
  }

  if (isTRUE(show_progress)) cat("Full-data imputations...\n")
  dat_full <- data.frame(y = y, X, check.names = FALSE)

  if (isTRUE(use_quickpred)) {
    qp <- quickpred_args; qp <- qp[!vapply(qp, is.null, logical(1))]
    pm_full <- do.call(mice::quickpred, c(list(data = dat_full), qp))
  } else {
    pm_full <- mice::make.predictorMatrix(dat_full)
  }

  if (type == "logistic" && is.numeric(dat_full$y) && all(na.omit(dat_full$y) %in% c(0,1))) {
    dat_full$y <- factor(dat_full$y, levels = c(0,1))
  }

  meth_full <- build_methods(dat_full)

  args_full <- modifyList(impute_args, list(
    data = dat_full,
    method = meth_full,
    predictorMatrix = pm_full
  ), keep.null = TRUE)
  if (is.null(args_full$m)) args_full$m <- m
  # seed for full-data imputations: offset far away but deterministic
  if (is.null(args_full$seed)) {
    args_full$seed <- seed_base_mice + 10^6
  } else {
    args_full$seed <- args_full$seed + 10^6
  }

  mids_full <- do.call(mice::mice, args_full)
  X_list <- vector("list", args_full$m)
  y_list <- vector("list", args_full$m)
  for (j in seq_len(args_full$m)) {
    comp_full <- mice::complete(mids_full, j)
    X_list[[j]] <- data.matrix(comp_full[, setdiff(names(comp_full), "y"), drop = FALSE])
    y_list[[j]] <- if (type == "logistic") as01(comp_full[["y"]]) else comp_full[["y"]]
  }

  if (isTRUE(show_progress))
    cat(sprintf("Boosting: k=%d, m=%d; mstop=%d...\n", k, m, mstop))

  res <- .cv_boost_core(
    X_train_list, y_train_list,
    X_val_list,   y_val_list,
    X_list,  y_list,
    ny, mstop, type, MIBoost, pool, pool_threshold,
    show_progress = show_progress,
    center = center
  )

  # classes
  class(res) <- c("booami_cv_raw", "booami_cv", class(res))

  # friendly metadata for prediction
  if (is.null(res$type)) res$type <- type

  if (is.null(res$center_means) && !is.null(res$final_fit) && !is.null(res$final_fit$center_means)) {
    res$center_means <- res$final_fit$center_means
  }
  if (is.null(res$center_means_list) && !is.null(res$final_fit) && !is.null(res$final_fit$center_means_list)) {
    res$center_means_list <- res$final_fit$center_means_list
  }

  pred_names <- colnames(X_list[[1L]])
  if (!is.null(res$final_model)) {
    if (!is.null(pred_names) &&
        length(res$final_model) == (length(pred_names) + 1L) &&
        is.null(names(res$final_model))) {
      names(res$final_model) <- c("(Intercept)", pred_names)
    }
  } else if (!is.null(res$final_models) && length(res$final_models)) {
    for (m_i in seq_along(res$final_models)) {
      fm <- res$final_models[[m_i]]
      if (!is.null(pred_names) &&
          length(fm) == (length(pred_names) + 1L) &&
          is.null(names(fm))) {
        names(fm) <- c("(Intercept)", pred_names)
        res$final_models[[m_i]] <- fm
      }
    }
  }

  if (isTRUE(return_full_imputations)) {
    res$full_imputations <- list(X = X_list, y = y_list)
  }

  # expose folds for reproducibility/auditing
  res$folds <- folds_id

  res
}












