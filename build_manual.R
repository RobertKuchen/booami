#!/usr/bin/env Rscript
build_manual <- function(pkg_dir = ".",
                         do_roxygen = TRUE,
                         build_data = TRUE,
                         copy_to_docs = TRUE,
                         quiet = FALSE) {
  msg <- function(...) if (!quiet) message(...)
  pkg_dir <- normalizePath(pkg_dir, winslash = "/", mustWork = TRUE)
  if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")
  if (!requireNamespace("roxygen2", quietly = TRUE)) install.packages("roxygen2")

  if (isTRUE(build_data)) {
    dr <- file.path(pkg_dir, "data-raw", "make_booami_sim.R")
    if (file.exists(dr)) {
      message("==> Generating dataset via data-raw/make_booami_sim.R ...")
      old <- getwd(); on.exit(setwd(old), add = TRUE)
      setwd(pkg_dir); sys.source(dr, envir = globalenv())
    }
  }

  if (isTRUE(do_roxygen)) { devtools::document(pkg_dir) }

  pdf <- tryCatch(devtools::build_manual(pkg_dir), error = function(e) NULL)
  if (is.null(pdf) || !file.exists(pdf)) {
    out <- file.path(pkg_dir, "booami-manual.pdf")
    system(sprintf("R CMD Rd2pdf %s -o %s", shQuote(pkg_dir), shQuote(out)))
    pdf <- out
  }
  if (isTRUE(copy_to_docs)) {
    docs <- file.path(pkg_dir, "docs"); dir.create(docs, showWarnings = FALSE, recursive = TRUE)
    file.copy(pdf, file.path(docs, basename(pdf)), overwrite = TRUE)
  }
  invisible(pdf)
}
if (identical(environment(), globalenv()) && !interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  target <- if (length(args)) args[[1]] else "."
  build_data <- TRUE
  if (length(args) >= 2) build_data <- as.logical(args[[2]])
  build_manual(target, do_roxygen = TRUE, build_data = build_data, copy_to_docs = TRUE)
}
