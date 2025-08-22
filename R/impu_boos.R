#' Component-Wise Gradient Boosting Across Multiply Imputed Datasets
#'
#' Applies component-wise gradient boosting to multiply imputed datasets.
#' Depending on the settings, either a separate model is reported for each
#' imputed dataset, or the M models are pooled to yield a single final model.
#' For pooling, one can choose the novel \emph{MIBoost} algorithm, which enforces
#' a uniform variable-selection scheme across all imputations, or the more
#' conventional ad-hoc approaches of estimate-averaging and
#' selection-frequency thresholding.
#'
#' @param X_list       List of length M; each element is an \eqn{n \times p} numeric
#'   predictor matrix from one imputed dataset.
#' @param y_list       List of length M; each element is a length-\eqn{n} numeric
#'   response vector from one imputed dataset.
#' @param X_list_val   Optional validation list (same structure as \code{X_list}).
#' @param y_list_val   Optional validation list (same structure as \code{y_list}).
#' @param ny Learning rate. Defaults to \code{0.1}.
#' @param mstop        Number of boosting iterations (default \code{250}).
#' @param type         Type of loss function. One of:
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
#'   in (0, 1)) proportion of the imputed datasets; coefficients of all other
#'   variables are set to zero. A value of \code{0} corresponds to
#'   estimate-averaging, while values \code{> 0} correspond to
#'   selection-frequency thresholding.
#' @param center One of \code{c("auto", "off", "force")}. Controls
#'   centering of \code{X} within each imputed dataset.
#'   With \code{"auto"} (recommended), centering is applied only if the training
#'   matrix is not already centered. With \code{"force"}, centering is always
#'   applied. With \code{"off"}, centering is skipped. If \code{X_list_val} is
#'   provided, validation sets are centered using the means from the
#'   corresponding training set.
#'
#' @return A list with elements:
#'   \itemize{
#'     \item \code{INT}: intercept(s). A scalar if \code{pool = TRUE}, otherwise
#'       a length-M vector.
#'     \item \code{BETA}: coefficient estimates. A length-p vector if
#'       \code{pool = TRUE}, otherwise an M \eqn{\times} p matrix.
#'     \item \code{CV_error}: vector of validation errors (if validation data
#'       were provided), otherwise \code{NULL}.
#'   }
#'
#' @details
#' This function supports \emph{MIBoost}, which enforces uniform variable
#' selection across multiply imputed datasets.
#' For full methodology, see Kuchen (2025).
#'
#' @references
#' Kuchen, R. (2025). \emph{MIBoost: A Gradient Boosting Algorithm for Variable
#' Selection After Multiple Imputation}. arXiv:2507.21807.
#' \doi{10.48550/arXiv.2507.21807} \url{https://arxiv.org/abs/2507.21807}.
#'
#' @examplesIf requireNamespace("mice", quietly = TRUE) && requireNamespace("miceadds", quietly = TRUE)
#' \donttest{
#'
#'   set.seed(123)
#'   utils::data(booami_sim)
#'
#'   M <- 2
#'   n <- nrow(booami_sim)
#'   x_cols <- grepl("^X\\d+$", names(booami_sim))
#'
#'   tr_idx <- sample(seq_len(n), floor(0.8 * n))
#'   dat_tr <- booami_sim[tr_idx, , drop = FALSE]
#'   dat_va <- booami_sim[-tr_idx, , drop = FALSE]
#'
#'   pm_tr <- mice::quickpred(dat_tr, method = "spearman",
#'                            mincor = 0.30, minpuc = 0.60)
#'
#'   imp_tr <- mice::mice(dat_tr, m = M, predictorMatrix = pm_tr,
#'                        maxit = 1, printFlag = FALSE)
#'   imp_va <- mice::mice.mids(imp_tr, newdata = dat_va, maxit = 1, printFlag = FALSE)
#'
#'   X_list      <- vector("list", M)
#'   y_list      <- vector("list", M)
#'   X_list_val  <- vector("list", M)
#'   y_list_val  <- vector("list", M)
#'   for (m in seq_len(M)) {
#'     tr_m <- mice::complete(imp_tr, m)
#'     va_m <- mice::complete(imp_va, m)
#'     X_list[[m]]     <- data.matrix(tr_m[, x_cols, drop = FALSE])
#'     y_list[[m]]     <- tr_m$y
#'     X_list_val[[m]] <- data.matrix(va_m[, x_cols, drop = FALSE])
#'     y_list_val[[m]] <- va_m$y
#'   }
#'
#'   fit <- impu_boost(
#'     X_list, y_list,
#'     X_list_val = X_list_val, y_list_val = y_list_val,
#'     ny = 0.1, mstop = 50, type = "gaussian",
#'     MIBoost = TRUE, pool = TRUE, center = "auto"
#'   )
#'
#'   which.min(fit$CV_error)
#'   head(fit$BETA)
#'   fit$INT
#' }
#'
#' \dontrun{
#' # Heavier demo (more imputations and iterations; for local runs)
#'
#'   set.seed(2025)
#'   utils::data(booami_sim)
#'
#'   M <- 10
#'   n <- nrow(booami_sim)
#'   x_cols <- grepl("^X\\d+$", names(booami_sim))
#'
#'   tr_idx <- sample(seq_len(n), floor(0.8 * n))
#'   dat_tr <- booami_sim[tr_idx, , drop = FALSE]
#'   dat_va <- booami_sim[-tr_idx, , drop = FALSE]
#'
#'   pm_tr <- mice::quickpred(dat_tr, method = "spearman",
#'                            mincor = 0.20, minpuc = 0.40)
#'
#'   imp_tr <- mice::mice(dat_tr, m = M, predictorMatrix = pm_tr,
#'                        maxit = 5, printFlag = TRUE)
#'   imp_va <- mice::mice.mids(imp_tr, newdata = dat_va, maxit = 1, printFlag = FALSE)
#'
#'   X_list      <- vector("list", M)
#'   y_list      <- vector("list", M)
#'   X_list_val  <- vector("list", M)
#'   y_list_val  <- vector("list", M)
#'   for (m in seq_len(M)) {
#'     tr_m <- mice::complete(imp_tr, m)
#'     va_m <- mice::complete(imp_va, m)
#'     X_list[[m]]     <- data.matrix(tr_m[, x_cols, drop = FALSE])
#'     y_list[[m]]     <- tr_m$y
#'     X_list_val[[m]] <- data.matrix(va_m[, x_cols, drop = FALSE])
#'     y_list_val[[m]] <- va_m$y
#'   }
#'
#'   fit_heavy <- impu_boost(
#'     X_list, y_list,
#'     X_list_val = X_list_val, y_list_val = y_list_val,
#'     ny = 0.1, mstop = 250, type = "gaussian",
#'     MIBoost = TRUE, pool = TRUE, center = "auto"
#'   )
#'   str(fit_heavy)
#' }
#'
#' @seealso \code{\link{simulate_booami_data}}, \code{\link{cv_boost_raw}}, \code{\link{cv_boost_imputed}}
#' @export
impu_boost <- function(X_list, y_list,
                       X_list_val = NULL, y_list_val = NULL,
                       ny = 0.1, mstop = 250,
                       type = c("gaussian", "logistic"),
                       MIBoost = TRUE, pool = TRUE,
                       pool_threshold = 0,
                       center = c("auto","force", "off")) {

  type   <- match.arg(type)
  center <- match.arg(center)

  # ---------- helpers ----------
  to_mat <- function(x) { if (!is.matrix(x)) x <- data.matrix(x); storage.mode(x) <- "double"; x }
  is_centered <- function(X) {
    mu  <- colMeans(X)
    sds <- apply(X, 2, function(v) suppressWarnings(stats::sd(v)))
    sds[!is.finite(sds) | sds <= 0] <- 1
    tol <- 1e-8 + 1e-6 * sds
    all(abs(mu) <= tol)
  }
  center_pair <- function(Xtr, Xva, mode) {
    Xtr <- to_mat(Xtr); Xva <- to_mat(Xva)
    if (mode == "off") return(list(train = Xtr, val = Xva, mu = NULL, centered = FALSE))
    need <- (mode == "force") || (!is_centered(Xtr))
    if (!need) return(list(train = Xtr, val = Xva, mu = NULL, centered = FALSE))
    mu <- colMeans(Xtr)
    list(train = sweep(Xtr, 2, mu, "-"),
         val   = sweep(Xva, 2, mu, "-"),
         mu = mu, centered = TRUE)
  }
  center_single <- function(X, mode) {
    X <- to_mat(X)
    if (mode == "off") return(list(X = X, mu = NULL, centered = FALSE))
    need <- (mode == "force") || (!is_centered(X))
    if (!need) return(list(X = X, mu = NULL, centered = FALSE))
    mu <- colMeans(X)
    list(X = sweep(X, 2, mu, "-"), mu = mu, centered = TRUE)
  }
  has_bad <- function(z) any(!is.finite(z))

  # ---------- input checks ----------
  M <- length(X_list); stopifnot(M == length(y_list))
  has_val <- !is.null(X_list_val) && !is.null(y_list_val)
  if (has_val) stopifnot(length(X_list_val) == M, length(y_list_val) == M)

  # basic NA/Inf checks (helps catch upstream issues early)
  for (m in seq_len(M)) {
    if (has_bad(as.numeric(y_list[[m]])) || has_bad(as.numeric(if (has_val) y_list_val[[m]] else 0)))
      stop(sprintf("Non-finite values in y for imputation %d.", m))
    if (has_bad(as.numeric(X_list[[m]])) || (has_val && has_bad(as.numeric(X_list_val[[m]]))))
      stop(sprintf("Non-finite values in X for imputation %d.", m))
  }

  # logistic guard: responses must be 0/1
  if (type == "logistic") {
    ok01 <- function(v) { v <- as.numeric(v); all(is.finite(v)) && all(v %in% c(0,1)) }
    if (!all(vapply(y_list, ok01, TRUE)))
      stop("For type='logistic', all y_list elements must be coded 0/1 (no NAs).")
    if (has_val && !all(vapply(y_list_val, ok01, TRUE)))
      stop("For type='logistic', all y_list_val elements must be coded 0/1 (no NAs).")
  }

  # track per-imputation training means actually used for centering
  mu_list <- vector("list", M)

  # ---------- centering per imputation ----------
  for (m in seq_len(M)) {
    if (has_val) {
      pair <- center_pair(X_list[[m]], X_list_val[[m]], mode = center)
      X_list[[m]]     <- pair$train
      X_list_val[[m]] <- pair$val
      mu_list[[m]]    <- if (pair$centered) pair$mu else NULL
    } else {
      cs <- center_single(X_list[[m]], mode = center)
      X_list[[m]]  <- cs$X
      mu_list[[m]] <- if (cs$centered) cs$mu else NULL
    }
    y_list[[m]] <- as.numeric(y_list[[m]])
    if (has_val) y_list_val[[m]] <- as.numeric(y_list_val[[m]])
  }

  p <- ncol(X_list[[1L]])

  # ---------- precompute base-learner operators ----------
  # use qr.solve for numerical stability; handles collinearity / constant columns
  BL_list <- vector("list", M)
  for (m in seq_len(M)) {
    BL_list[[m]] <- vector("list", p)
    X <- X_list[[m]]
    for (r in seq_len(p)) {
      x <- cbind(1, X[, r])
      # computes (X'X)^(-1) X' via QR-based solve
      BL_list[[m]][[r]] <- qr.solve(crossprod(x), t(x))
    }
  }

  # ---------- initialize ----------
  BETA <- matrix(0, nrow = M, ncol = p)
  INT  <- numeric(M)
  OOS_CV <- if (has_val) numeric(mstop) else NULL

  # ---------- boosting path ----------
  for (t in seq_len(mstop)) {
    Est_Inter <- matrix(0, nrow = M, ncol = p)
    Est_Coef  <- matrix(0, nrow = M, ncol = p)
    RSS       <- matrix(0, nrow = M, ncol = p)

    for (m in seq_len(M)) {
      X   <- X_list[[m]]
      lp  <- as.vector(INT[m] + X %*% BETA[m, ])
      eta <- if (type == "logistic") 1/(1 + exp(-lp)) else lp
      u   <- y_list[[m]] - eta

      for (r in seq_len(p)) {
        fit <- BL_list[[m]][[r]] %*% u
        Est_Inter[m, r] <- fit[1]
        Est_Coef[m, r]  <- fit[2]
        RSS[m, r] <- sum((u - cbind(1, X[, r]) %*% fit)^2)
      }

      if (!MIBoost) {
        best <- which.min(RSS[m, ])
        BETA[m, best] <- BETA[m, best] + ny * Est_Coef[m, best]
        INT[m]        <- INT[m]        + ny * Est_Inter[m, best]
      }
    }

    if (MIBoost) {
      best <- which.min(colMeans(RSS))
      INT          <- INT          + ny * Est_Inter[, best]
      BETA[, best] <- BETA[, best] + ny * Est_Coef[, best]
    }

    # validation loss
    if (has_val) {
      OOS_loss <- numeric(M)
      if (pool) {
        BETA_pool <- BETA
        if (!MIBoost && is.numeric(pool_threshold) && pool_threshold > 0) {
          keep_prop <- colMeans(BETA_pool != 0)
          drop_cols <- which(keep_prop < pool_threshold)
          if (length(drop_cols)) BETA_pool[, drop_cols] <- 0
        }
        BETA_val <- colMeans(BETA_pool)
        INT_val  <- mean(INT)

        for (m in seq_len(M)) {
          Xv <- X_list_val[[m]]; yv <- y_list_val[[m]]
          if (type == "gaussian") {
            OOS_loss[m] <- mean((yv - INT_val - Xv %*% BETA_val)^2)
          } else {
            p_val <- 1 / (1 + exp(-(INT_val + Xv %*% BETA_val)))
            p_val <- pmin(pmax(p_val, 1e-8), 1 - 1e-8)
            OOS_loss[m] <- -2 * mean(yv * log(p_val) + (1 - yv) * log(1 - p_val))
          }
        }
      } else {
        for (m in seq_len(M)) {
          Xv <- X_list_val[[m]]; yv <- y_list_val[[m]]
          if (type == "gaussian") {
            OOS_loss[m] <- mean((yv - INT[m] - Xv %*% BETA[m, ])^2)
          } else {
            p_val <- 1 / (1 + exp(-(INT[m] + Xv %*% BETA[m, ])))
            p_val <- pmin(pmax(p_val, 1e-8), 1 - 1e-8)
            OOS_loss[m] <- -2 * mean(yv * log(p_val) + (1 - yv) * log(1 - p_val))
          }
        }
      }
      OOS_CV[t] <- mean(OOS_loss)
    }
  }

  # ---------- final pooling ----------
  if (pool) {
    BETA_mat <- BETA
    if (!MIBoost && is.numeric(pool_threshold) && pool_threshold > 0) {
      keep_prop <- colMeans(BETA_mat != 0)
      drop_cols <- which(keep_prop < pool_threshold)
      if (length(drop_cols)) BETA_mat[, drop_cols] <- 0
    }
    BETA <- colMeans(BETA_mat)
    INT  <- mean(INT)
  }

  # ---------- compute returned center means ----------
  center_info <- NULL
  if (pool) {
    any_mu <- which(vapply(mu_list, Negate(is.null), logical(1)))
    if (length(any_mu)) {
      proto <- mu_list[[ any_mu[1] ]]
      MU <- do.call(cbind, lapply(seq_len(M), function(m) {
        if (is.null(mu_list[[m]])) {
          stats::setNames(rep(0, length(proto)), names(proto))
        } else {
          mu_list[[m]][names(proto)]
        }
      }))
      center_info <- rowMeans(MU)
    } else {
      center_info <- NULL
    }
  } else {
    center_info <- mu_list
  }

  # ---------- assemble result ----------
  res <- list(
    INT = INT,
    BETA = BETA,
    CV_error = OOS_CV,
    type = type
  )
  if (pool) {
    if (!is.null(center_info)) res$center_means <- center_info
    class(res) <- c("booami_pooled", "booami_fit", class(res))
  } else {
    res$center_means_list <- center_info
    class(res) <- c("booami_multi", "booami_fit", class(res))
  }

  res
}




