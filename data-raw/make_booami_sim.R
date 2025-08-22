# data-raw/make_booami_sim.R  ----
# Always use the dev version of simulate_booami_data
if (requireNamespace("pkgload", quietly = TRUE)) {
  pkgload::load_all(export_all = TRUE)
  pkg <- pkgload::pkg_name()
  simulate_booami_data <- getFromNamespace("simulate_booami_data", pkg)
} else {
  # Fallback: source all R/ files so the function exists in this session
  message("pkgload not available; sourcing R/ files")
  r_files <- list.files("R", pattern = "[.]R$", full.names = TRUE)
  for (f in r_files) sys.source(f, envir = environment())
}

# Sanity check: make sure we have the *new* signature
needed <- c("miss_prop", "keep_mar_drivers")
have   <- names(formals(simulate_booami_data))
if (!all(needed %in% have)) {
  stop("simulate_booami_data found, but missing args: ",
       paste(setdiff(needed, have), collapse = ", "),
       "\nFound formals: ", paste(have, collapse = ", "))
}

set.seed(123)
sim <- simulate_booami_data(
  n = 300, p = 25, p_inf = 5, rho = 0.3,
  type = "gaussian",
  beta_range = c(1, 2),
  intercept = 1,
  corr_structure = "all_ar1",
  rho_noise = NULL,
  noise_sd = 1,
  miss = "MAR",
  miss_prop = 0.25,
  mar_drivers = c(1, 2, 3),
  gamma_vec = NULL,
  calibrate_mar = FALSE,
  mar_scale = TRUE,
  keep_observed = integer(0),
  jitter_sd = 0.25,
  keep_mar_drivers = TRUE
)

booami_sim <- sim$data
if (requireNamespace("usethis", quietly = TRUE)) {
  usethis::use_data(booami_sim, overwrite = TRUE, compress = "xz")
} else {
  dir.create("data", showWarnings = FALSE)
  save(booami_sim, file = file.path("data", "booami_sim.rda"), compress = "xz")
}

