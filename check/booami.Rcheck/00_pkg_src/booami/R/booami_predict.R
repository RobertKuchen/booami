#' Predict with booami models
#'
#' Minimal, dependency-free predictor for models fitted by
#' \code{\link{cv_boost_raw}}, \code{\link{cv_boost_imputed}}, or a
#' *pooled* \code{\link{impu_boost}} fit. Supports Gaussian (identity)
#' and logistic (logit) models, returning either the linear predictor
#' or, for logistic, predicted probabilities.
#'
#' @param object A fit returned by \code{cv_boost_raw()}, \code{cv_boost_imputed()},
#'   or a \emph{pooled} \code{impu_boost()} (i.e., \code{pool = TRUE} so that
#'   \code{$BETA} is a length-\eqn{p} vector and \code{$INT} is a scalar).
#' @param X_new New data (matrix or data.frame) with the same \eqn{p} predictors
#'   the model was trained on. If column names are present in the model,
#'   \code{X_new} will be aligned by name; otherwise it must be in the same order.
#' @param family Model family; one of \code{c("gaussian","logistic")}. If
#'   \code{NULL}, the function tries to infer from \code{object$type} or
#'   attributes; otherwise defaults to \code{"gaussian"}.
#' @param type Prediction type; one of \code{c("response","link")}. For
#'   \code{"gaussian"}, both are identical. For \code{"logistic"},
#'   \code{"response"} returns probabilities via the inverse-logit.
#' @param center_means Optional numeric vector of length \eqn{p} with training
#'   means used to center predictors during fitting. If provided, \code{X_new}
#'   is centered as \code{X_new - center_means} before prediction. If the model
#'   stores means by name, pass a named vector whose names match predictor names.
#'
#' @details
#' This function is deterministic and involves no random number generation.
#' Coefficients are extracted from either \code{$final_model} (intercept first,
#' then coefficients) or from \code{$INT}+\code{$BETA} (pooled \code{impu_boost}).
#' If \code{X_new} has column names and the model has named coefficients, columns
#' are aligned by name; otherwise they are used in order.
#'
#' If your training pipeline centered covariates (e.g., \code{center = "auto"}),
#' providing the same \code{center_means} here yields numerically consistent
#' predictions. If not supplied but \code{object$center_means} exists, it will
#' be used automatically. If both are supplied, the explicit \code{center_means}
#' argument takes precedence.
#'
#' @return A numeric vector of predictions (length \code{nrow(X_new)}). If
#'   \code{X_new} has row names, they are propagated to the returned vector.
#'
#' @examplesIf requireNamespace("mice", quietly = TRUE) && requireNamespace("miceadds", quietly = TRUE)
#' \donttest{
#' # 1) Fit on data WITH missing values
#' set.seed(123)
#' sim_tr <- simulate_booami_data(
#'   n = 120, p = 12, p_inf = 3,
#'   type = "gaussian",
#'   miss = "MAR", miss_prop = 0.20
#' )
#' X_tr <- sim_tr$data[, 1:12]
#' y_tr <- sim_tr$data$y
#'
#' fit <- cv_boost_raw(
#'   X_tr, y_tr,
#'   k = 2, mstop = 50, seed = 123,
#'   impute_args    = list(m = 2, maxit = 1, printFlag = FALSE, seed = 1),
#'   quickpred_args = list(method = "spearman", mincor = 0.30, minpuc = 0.60),
#'   show_progress  = FALSE
#' )
#'
#' # 2) Predict on a separate data set WITHOUT missing values (same p)
#' sim_new <- simulate_booami_data(
#'   n = 5, p = 12, p_inf = 3,
#'   type = "gaussian",
#'   miss = "MCAR", miss_prop = 0   # <- complete data with existing API
#' )
#' X_new <- sim_new$data[, 1:12, drop = FALSE]
#'
#' preds <- booami_predict(fit, X_new = X_new, family = "gaussian", type = "response")
#' round(preds, 3)
#' }
#'
#' @seealso \code{\link{cv_boost_raw}}, \code{\link{cv_boost_imputed}}, \code{\link{impu_boost}}
#' @export
booami_predict <- function(object, X_new,
                           family = NULL,
                           type   = c("response", "link"),
                           center_means = NULL) {
  # --- parse args ------------------------------------------------------------
  type <- match.arg(type)

  infer_family <- function(obj) {
    cand <- NULL
    if (!is.null(obj$type)) cand <- obj$type
    if (is.null(cand)) cand <- attr(obj, "type")
    if (is.null(cand)) cand <- attr(obj, "family")
    if (!is.null(cand)) {
      cand <- tolower(as.character(cand))
      if (cand %in% c("gaussian", "logistic")) return(cand)
      if (cand %in% c("binomial", "bernoulli")) return("logistic")
    }
    "gaussian"
  }
  if (is.null(family)) family <- infer_family(object)
  family <- match.arg(family, c("gaussian", "logistic"))

  # --- inherit saved training means if available (caller takes precedence) ---
  if (is.null(center_means)) center_means <- object$center_means

  # --- extract coefficients --------------------------------------------------
  intercept <- NULL
  beta <- NULL

  if (!is.null(object$final_model)) {
    # cv_boost_*: final_model is length 1+p (intercept first)
    cf <- as.numeric(object$final_model)
    if (length(cf) < 2L)
      stop("object$final_model must contain intercept + coefficients.")
    intercept <- unname(cf[1L])
    beta <- cf[-1L]
    nm <- names(object$final_model)
    if (!is.null(nm) && length(nm) == (length(beta) + 1L)) {
      names(beta) <- nm[-1L]
    }
  } else if (!is.null(object$BETA) && !is.null(object$INT)) {
    # impu_boost pooled: BETA is vector, INT is scalar
    if (is.matrix(object$BETA)) {
      stop("Object appears to be from unpooled impu_boost (BETA is M x p). ",
           "Use pool = TRUE at fitting time, or average per-imputation predictions manually.")
    }
    intercept <- as.numeric(object$INT)
    beta <- as.numeric(object$BETA)
    names(beta) <- names(object$BETA)
  } else {
    stop("Unsupported object: expected $final_model or pooled $INT + $BETA.")
  }

  # --- prepare X_new ---------------------------------------------------------
  X_new <- as.matrix(X_new)
  storage.mode(X_new) <- "double"

  # align by names if available
  if (!is.null(names(beta)) && length(names(beta)) && all(!is.na(names(beta)))) {
    missing_cols <- setdiff(names(beta), colnames(X_new))
    if (length(missing_cols)) {
      stop("X_new is missing required columns: ",
           paste(missing_cols, collapse = ", "))
    }
    X_new <- X_new[, names(beta), drop = FALSE]
  } else if (ncol(X_new) != length(beta)) {
    stop("X_new has ", ncol(X_new), " columns but model has ",
         length(beta), " coefficients.")
  }

  # optional centering using provided/learned means
  if (!is.null(center_means)) {
    cm <- center_means
    if (!is.null(names(beta)) && !is.null(names(center_means))) {
      if (!all(names(beta) %in% names(center_means))) {
        stop("center_means must be a named vector covering all predictor names.")
      }
      cm <- center_means[names(beta)]
    } else if (length(center_means) != length(beta)) {
      stop("center_means must have length = number of predictors.")
    }
    X_use <- sweep(X_new, 2L, cm, FUN = "-")
  } else {
    X_use <- X_new
  }

  # --- linear predictor and transform ---------------------------------------
  eta <- drop(intercept + X_use %*% beta)
  out <- if (family == "logistic" && type == "response") stats::plogis(eta) else eta

  # preserve row names if present
  rn <- rownames(X_new)
  if (!is.null(rn)) names(out) <- rn

  out
}
