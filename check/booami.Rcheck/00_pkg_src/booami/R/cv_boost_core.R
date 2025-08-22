#' @keywords internal
.cv_boost_core <- function(
    X_train_list, y_train_list,
    X_val_list,   y_val_list,
    X, y,
    ny = 0.1, mstop = 250, type = c("gaussian", "logistic"),
    MIBoost = TRUE, pool = TRUE, pool_threshold = 0,
    show_progress = TRUE,
    center = "auto"
) {
  type   <- match.arg(type)
  center <- match.arg(center, c("auto","off","force"))

  k <- length(y_train_list)
  stopifnot(length(X_train_list) == k,
            length(X_val_list)   == k,
            length(y_val_list)   == k)

  # p only used for sanity; prediction-time naming handled below
  X1 <- X_train_list[[1]][[1]]
  if (!is.matrix(X1)) X1 <- data.matrix(X1)
  p <- ncol(X1); stopifnot(is.finite(p), p >= 1)

  if (isTRUE(show_progress)) {
    cat(sprintf("Cross-validation over %d folds...\n", k));
    utils::flush.console()
  }

  t0 <- proc.time()
  res_fold <- vector("list", k)
  for (cv in seq_len(k)) {
    if (isTRUE(show_progress)) cat(sprintf("  - Fold %d/%d\n", cv, k))
    res_fold[[cv]] <- impu_boost(
      X_list      = X_train_list[[cv]],
      y_list      = y_train_list[[cv]],
      X_list_val  = X_val_list[[cv]],
      y_list_val  = y_val_list[[cv]],
      ny = ny, mstop = mstop, type = type,
      MIBoost = MIBoost, pool = pool, pool_threshold = pool_threshold,
      center = center
    )$CV_error
  }
  cv_time <- as.numeric((proc.time() - t0)["elapsed"])
  mean_cv_error <- Reduce("+", res_fold) / k
  best_mstop <- which.min(mean_cv_error)
  if (isTRUE(show_progress)) cat(sprintf("CV finished in %.2fs. Best mstop = %d\n", cv_time, best_mstop))
  if (isTRUE(show_progress)) cat("Fitting final model on full imputations...\n")

  # Final pooled (or unpooled) fit on full imputations, at chosen mstop
  fit <- impu_boost(
    X_list = X, y_list = y,
    ny = ny, mstop = best_mstop, type = type,
    MIBoost = MIBoost, pool = pool, pool_threshold = pool_threshold,
    center = center
  )

  # Assemble return object
  out <- list(
    CV_error   = mean_cv_error,
    CV_error_per_fold = res_fold,
    best_mstop = best_mstop,
    type       = type,
    final_fit  = fit
  )

  # Name predictors from the first full-imputation design matrix
  pred_names <- colnames(if (is.matrix(X[[1]])) X[[1]] else data.matrix(X[[1]]))

  if (isTRUE(pool)) {
    fm <- c(fit$INT, fit$BETA)
    if (!is.null(pred_names) && length(fm) == length(pred_names) + 1L) {
      names(fm) <- c("(Intercept)", pred_names)
    }
    out$final_model <- fm
    if (!is.null(fit$center_means)) out$center_means <- fit$center_means
  } else {
    M <- length(fit$INT)
    fm_list <- vector("list", M)
    for (m in seq_len(M)) {
      fm <- c(fit$INT[m], fit$BETA[m, ])
      if (!is.null(pred_names) && length(fm) == length(pred_names) + 1L) {
        names(fm) <- c("(Intercept)", pred_names)
      }
      fm_list[[m]] <- fm
    }
    out$final_models <- fm_list
    if (!is.null(fit$center_means_list)) out$center_means_list <- fit$center_means_list
  }

  out
}

