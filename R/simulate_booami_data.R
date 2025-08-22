#' Simulate a Booami Example Dataset with Missing Values
#'
#' Generates a dataset with \eqn{p} predictors, of which the first \code{p_inf}
#' are informative. Predictors are drawn from a multivariate normal with a chosen
#' correlation structure, and the outcome can be continuous (\code{type = "gaussian"})
#' or binary (\code{type = "logistic"}). Missing values are introduced via MAR or MCAR.
#'
#' \strong{Correlation structures:}
#' \itemize{
#'   \item \code{"all_ar1"}: AR(1) correlation with parameter \code{rho} across all \eqn{p} predictors.
#'   \item \code{"informative_cs"}: compound symmetry (exchangeable) within the first \code{p_inf}
#'         predictors with parameter \code{rho}; others independent.
#'   \item \code{"blockdiag"}: block-diagonal AR(1): the informative block (size \code{p_inf}) has AR(1) with \code{rho};
#'         the noise block (size \code{p - p_inf}) has AR(1) with \code{rho_noise} (defaults to \code{rho}).
#'   \item \code{"none"}: independent predictors.
#' }
#'
#' \strong{Missingness:}
#' \itemize{
#'   \item \code{"MAR"}: for each row, a logit missingness score is computed from the
#'         selected MAR drivers (see \code{mar_drivers}, \code{gamma_vec}, \code{mar_scale});
#'         an intercept is set via \code{calibrate_mar} to target the proportion \code{miss_prop}
#'         (otherwise \code{qlogis(miss_prop)}),
#'         and per-row jitter \eqn{N(0, \code{jitter_sd})} adds heterogeneity. The resulting probability
#'         is used to mask predictors (except those in \code{keep_observed} and—if \code{keep_mar_drivers = TRUE}—the drivers themselves).
#'         For \code{type = "gaussian"} only, \code{y} is also subject to the same missingness mechanism.
#'   \item \code{"MCAR"}: each predictor (except those in \code{keep_observed}) is masked independently with probability \code{miss_prop}.
#'         For \code{type = "gaussian"} only, \code{y} is also masked MCAR with probability \code{miss_prop}.
#' }
#'
#' \emph{Note:} In the simulation, missingness probabilities are computed using the
#' fully observed latent covariates before masking. From an analyst’s perspective after
#' masking, allowing the MAR drivers themselves to be missing makes missingness depend on
#' unobserved values—i.e., effectively non-ignorable (MNAR). Setting
#' \code{keep_mar_drivers = TRUE} keeps those drivers observed and yields a MAR mechanism.
#'
#' @param n Number of observations (default \code{300}).
#' @param p Total number of predictors (default \code{25}).
#' @param p_inf Number of informative predictors (default \code{5}); must satisfy \code{p_inf <= p}.
#' @param rho Correlation parameter (interpretation depends on \code{corr_structure}).
#' @param type Either \code{"gaussian"} or \code{"logistic"} (default \code{"gaussian"}).
#' @param beta_range Length-2 numeric; coefficients for the first \code{p_inf} informative predictors are drawn
#'   i.i.d. \code{Uniform(beta_range[1], beta_range[2])}.
#' @param intercept Intercept added to the linear predictor (default \code{1}).
#' @param corr_structure One of \code{"all_ar1"}, \code{"informative_cs"}, \code{"blockdiag"}, \code{"none"}.
#' @param rho_noise Optional correlation for the noise block when \code{corr_structure = "blockdiag"}
#'   (defaults to \code{rho}).
#' @param noise_sd Std. dev. of Gaussian noise added to \code{y} when \code{type = "gaussian"} (default \code{1});
#'   ignored for \code{type = "logistic"}.
#' @param miss Missingness mechanism: \code{"MAR"} or \code{"MCAR"} (default \code{"MAR"}).
#' @param miss_prop Target marginal missingness proportion (default \code{0.25}).
#' @param mar_drivers Indices of predictors that drive MAR (default \code{c(1, 2, 3)}). Must lie within \code{1..p}.
#'   (Out-of-range indices are ignored; an empty set is not allowed.)
#' @param gamma_vec Coefficients for MAR drivers; length must equal the number of MAR drivers actually used
#'   (i.e., \code{length(mar_drivers)} after restricting to \code{1..p}). If \code{NULL}, heuristic defaults are used
#'   (starting from \code{c(0.5, -0.35, 0.15)} as available).
#' @param calibrate_mar If \code{TRUE}, calibrates the MAR intercept by root-finding so that the average missingness
#'   matches \code{miss_prop}. If \code{FALSE}, uses \code{qlogis(miss_prop)}.
#' @param mar_scale If \code{TRUE} (default), standardize MAR drivers before applying \code{gamma_vec}.
#' @param keep_observed Indices of predictors kept fully observed (values outside \code{1..p} are ignored).
#' @param jitter_sd Standard deviation of the per-row jitter added to the MAR logit to induce heterogeneity
#'   (default \code{0.25}).
#' @param keep_mar_drivers Logical; if \code{TRUE} (default), predictors in
#'   \code{mar_drivers} are kept fully observed under MAR so that missingness
#'   depends only on observed covariates (MAR). If \code{FALSE}, those drivers
#'   may be masked as well, making the mechanism effectively non-ignorable
#'   (MNAR) for variables whose missingness depends on them.
#'
#' @return A \code{list} with elements:
#' \itemize{
#'   \item \code{data}: \code{data.frame} with columns \code{X1..Xp} and \code{y}, containing \code{NA}s per the missingness mechanism.
#'   \item \code{beta}: numeric length-\code{p} vector of true coefficients (non-zeros in the first \code{p_inf} positions).
#'   \item \code{informative}: integer vector \code{1:p_inf}.
#'   \item \code{type}: character, outcome type (\code{"gaussian"} or \code{"logistic"}).
#'   \item \code{intercept}: numeric intercept used.
#' }
#'
#' The \code{data} element additionally carries attributes:
#' \code{"true_beta"}, \code{"informative"},
#' \code{"type"}, \code{"corr_structure"}, \code{"rho"}, \code{"rho_noise"} (if set),
#' \code{"intercept"}, \code{"noise_sd"} (Gaussian; \code{NA} otherwise), \code{"mar_scale"},
#' and \code{"keep_mar_drivers"}.
#'
#' @section Reproducing the shipped dataset \code{booami_sim}:
#' \preformatted{
#' set.seed(123)
#' sim <- simulate_booami_data(
#'   n = 300, p = 25, p_inf = 5, rho = 0.3,
#'   type = "gaussian", beta_range = c(1, 2), intercept = 1,
#'   corr_structure = "all_ar1", rho_noise = NULL, noise_sd = 1,
#'   miss = "MAR", miss_prop = 0.25,
#'   mar_drivers = c(1, 2, 3), gamma_vec = NULL,
#'   calibrate_mar = FALSE, mar_scale = TRUE,
#'   keep_observed = integer(0), jitter_sd = 0.25,
#'   keep_mar_drivers = TRUE
#' )
#' booami_sim <- sim$data
#' }
#'
#' @examples
#' set.seed(42)
#' sim <- simulate_booami_data(
#'   n = 200, p = 15, p_inf = 4, rho = 0.25,
#'   type = "gaussian", miss = "MAR", miss_prop = 0.20
#' )
#' d <- sim$data
#' dim(d)
#' mean(colSums(is.na(d)) > 0)    # fraction of columns with any NAs
#' head(attr(d, "true_beta"))
#' attr(d, "informative")
#'
#' # Example with block-diagonal correlation and protected MAR drivers
#' sim2 <- simulate_booami_data(
#'   n = 150, p = 12, p_inf = 3, rho = 0.40, rho_noise = 0.10,
#'   corr_structure = "blockdiag", miss = "MAR", miss_prop = 0.30,
#'   mar_drivers = c(1, 2), keep_mar_drivers = TRUE
#' )
#' colSums(is.na(sim2$data))[1:4]
#'
#' # Binary outcome example
#' sim3 <- simulate_booami_data(
#'   n = 100, p = 10, p_inf = 2, rho = 0.2,
#'   type = "logistic", miss = "MCAR", miss_prop = 0.15
#' )
#' table(sim3$data$y, useNA = "ifany")
#'
#' \donttest{
#' utils::data(booami_sim)
#' dim(booami_sim)
#' head(attr(booami_sim, "true_beta"))
#' attr(booami_sim, "informative")
#' }
#'
#' @seealso \code{\link{booami_sim}}, \code{\link{cv_boost_raw}},
#'   \code{\link{cv_boost_imputed}}, \code{\link{impu_boost}}
#' @export
simulate_booami_data <- function(
    n = 300, p = 25, p_inf = 5, rho = 0.3,
    type = c("gaussian","logistic"),
    beta_range = c(1, 2),
    intercept = 1,
    corr_structure = c("all_ar1","informative_cs","blockdiag","none"),
    rho_noise = NULL,
    noise_sd = 1,
    miss = c("MAR","MCAR"),
    miss_prop = 0.25,
    mar_drivers = c(1, 2, 3),
    gamma_vec = NULL,
    calibrate_mar = FALSE,
    mar_scale = TRUE,
    keep_observed = integer(0),
    jitter_sd = 0.25,
    keep_mar_drivers = TRUE   # <- final, canonical name
){
  type <- match.arg(type)
  corr_structure <- match.arg(corr_structure)
  miss <- match.arg(miss)
  stopifnot(p_inf <= p)
  stopifnot(is.numeric(beta_range) && length(beta_range) == 2 && beta_range[1] < beta_range[2])
  stopifnot(is.numeric(noise_sd) && noise_sd >= 0)

  ar1 <- function(pp, r) r^abs(outer(seq_len(pp), seq_len(pp), "-"))
  cs  <- function(pp, r) { S <- matrix(r, pp, pp); diag(S) <- 1; S }

  # Covariance
  if (corr_structure == "all_ar1") {
    Sigma <- ar1(p, rho)
  } else if (corr_structure == "informative_cs") {
    Sigma <- diag(p); if (p_inf > 0) Sigma[1:p_inf, 1:p_inf] <- cs(p_inf, rho)
  } else if (corr_structure == "blockdiag") {
    Sigma <- diag(p)
    if (p_inf > 0) Sigma[1:p_inf, 1:p_inf] <- ar1(p_inf, rho)
    if (p_inf < p) {
      rn <- if (is.null(rho_noise)) rho else rho_noise
      Sigma[(p_inf+1):p, (p_inf+1):p] <- ar1(p - p_inf, rn)
    }
  } else {
    Sigma <- diag(p)
  }

  # Predictors
  X <- MASS::mvrnorm(n = n, mu = rep(0, p), Sigma = Sigma)
  colnames(X) <- paste0("X", seq_len(p))

  # Coefficients
  beta <- numeric(p)
  beta[seq_len(p_inf)] <- stats::runif(p_inf, min = beta_range[1], max = beta_range[2])

  # Outcome
  eta <- as.vector(intercept + X %*% beta)
  if (type == "gaussian") {
    y <- eta + stats::rnorm(n, sd = noise_sd)
  } else {
    py <- 1/(1 + exp(-eta))
    y <- stats::rbinom(n, 1, py)
  }
  df <- data.frame(X, y = y, check.names = FALSE)

  # Missingness
  if (miss == "MAR") {
    drivers <- mar_drivers[mar_drivers >= 1 & mar_drivers <= p]
    if (length(drivers) == 0) stop("mar_drivers indices must be within 1..p.")
    Z <- X[, drivers, drop = FALSE]
    if (mar_scale) {
      Z <- scale(Z); if (is.null(dim(Z))) Z <- matrix(Z, ncol = 1)
    } else {
      Z <- as.matrix(Z)
    }
    if (is.null(gamma_vec)) {
      gamma_vec <- rep(0, ncol(Z))
      base <- c(0.5, -0.35, 0.15)
      gamma_vec[seq_along(base)[seq_len(min(length(base), ncol(Z)))]] <- base[seq_len(min(length(base), ncol(Z)))]
    } else if (length(gamma_vec) != ncol(Z)) {
      stop("gamma_vec length must equal number of MAR drivers.")
    }

    lp_core <- as.vector(Z %*% gamma_vec)
    alpha <- if (isTRUE(calibrate_mar)) {
      f <- function(a) mean(stats::plogis(a + lp_core)) - miss_prop
      stats::uniroot(f, interval = c(-10, 10))$root
    } else {
      stats::qlogis(miss_prop)
    }
    prob_miss <- plogis(alpha + lp_core + stats::rnorm(n, sd = jitter_sd))

    df_miss <- df

    # Keep: user-specified + (drivers if keep_mar_drivers)
    keep <- intersect(keep_observed, seq_len(p))
    if (isTRUE(keep_mar_drivers)) keep <- union(keep, drivers)

    for (j in seq_len(p)) {
      if (j %in% keep) next
      mask <- stats::rbinom(n, 1, prob_miss) == 1
      df_miss[mask, paste0("X", j)] <- NA
    }
    if (type == "gaussian") {
      mask_y <- stats::rbinom(n, 1, prob_miss) == 1
      df_miss[mask_y, "y"] <- NA
    }
    df <- df_miss

  } else { # MCAR
    df_miss <- df
    keep <- intersect(keep_observed, seq_len(p))
    for (j in seq_len(p)) {
      if (j %in% keep) next
      mask <- stats::rbinom(n, 1, miss_prop) == 1
      df_miss[mask, paste0("X", j)] <- NA
    }
    if (type == "gaussian") {
      df_miss[stats::rbinom(n, 1, miss_prop) == 1, "y"] <- NA
    }
    df <- df_miss
  }

  # Attributes
  attr(df, "true_beta") <- beta
  attr(df, "informative") <- seq_len(p_inf)
  attr(df, "type") <- type
  attr(df, "corr_structure") <- corr_structure
  attr(df, "rho") <- rho
  if (!is.null(rho_noise)) attr(df, "rho_noise") <- rho_noise
  attr(df, "intercept") <- intercept
  attr(df, "noise_sd") <- if (type == "gaussian") noise_sd else NA_real_
  attr(df, "mar_scale") <- mar_scale
  attr(df, "keep_mar_drivers") <- keep_mar_drivers

  list(
    data = df,
    beta = beta,
    informative = seq_len(p_inf),
    type = type,
    intercept = intercept
  )
}








