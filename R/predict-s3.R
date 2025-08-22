# R/predict-s3.R

#' Predict from booami objects
#'
#' Predict responses (link or response scale) from fitted booami models.
#'
#' @param object A fitted booami object. One of:
#'   \itemize{
#'     \item \code{"booami_cv"} — cross-validated model object.
#'     \item \code{"booami_pooled"} — pooled fit from \code{impu_boost(..., pool = TRUE)}.
#'     \item \code{"booami_multi"} — unpooled fit from \code{impu_boost(..., pool = FALSE)}.
#'   }
#' @param newdata A data.frame or matrix of predictors (same columns/order as training).
#' @param type Either \code{"link"} for the linear predictor, or \code{"response"} for
#'   mean/probability (Gaussian/logistic respectively).
#' @param ... Passed to \code{\link{booami_predict}}. For \code{"booami_multi"}, you may
#'   use \code{aggregate = "mean"|"median"|NULL} and/or \code{which_m = <index>} to
#'   control how predictions are aggregated across imputations.
#'
#' @return A numeric vector of predictions.
#'
#' @seealso \code{\link{booami_predict}}
#' @name predict.booami
NULL

#' @rdname predict.booami
#' @method predict booami_cv
#' @export
predict.booami_cv <- function(object, newdata, type = c("link", "response"), ...) {
  type <- match.arg(type)
  booami_predict(object = object, X_new = newdata, type = type, ...)
}

#' @rdname predict.booami
#' @method predict booami_pooled
#' @export
predict.booami_pooled <- function(object, newdata, type = c("link", "response"), ...) {
  type <- match.arg(type)
  booami_predict(object = object, X_new = newdata, type = type, ...)
}

#' @rdname predict.booami
#' @method predict booami_multi
#' @export
predict.booami_multi <- function(object, newdata, type = c("link", "response"), ...) {
  type <- match.arg(type)
  booami_predict(object = object, X_new = newdata, type = type, ...)
}
