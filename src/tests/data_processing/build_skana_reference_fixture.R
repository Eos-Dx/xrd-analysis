#!/usr/bin/env Rscript

# Optional helper to generate a parity fixture from SK-Ana sample data.
# Output: src/tests/data_processing/fixtures/skana/parity_fixture.npz

suppressPackageStartupMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Please install jsonlite: install.packages('jsonlite')")
  }
  if (!requireNamespace("nnls", quietly = TRUE)) {
    stop("Please install nnls: install.packages('nnls')")
  }
})

# helper for string concatenation compatible with base R
`%+%` <- function(a, b) paste0(a, b)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_dir <- if (length(file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = TRUE))
} else {
  getwd()
}
repo_root <- normalizePath(file.path(script_dir, "..", "..", ".."), mustWork = TRUE)
fixture_dir <- file.path(repo_root, "src", "tests", "data_processing", "fixtures", "skana")
if (!dir.exists(fixture_dir)) dir.create(fixture_dir, recursive = TRUE)

skana_data <- "~/dev/SK-Ana/data/data.csv"
skana_data <- path.expand(skana_data)
if (!file.exists(skana_data)) {
  stop(sprintf("SK-Ana sample data not found: %s", skana_data))
}

raw <- as.matrix(read.csv(skana_data, header = FALSE, check.names = FALSE))
wavelength <- as.numeric(raw[1, -1])
delay <- as.numeric(raw[-1, 1])
mat <- raw[-1, -1]
storage.mode(mat) <- "double"

# Keep a bounded block so fixture size stays manageable.
n_delay <- min(120, nrow(mat))
n_wav <- min(200, ncol(mat))
mat <- mat[1:n_delay, 1:n_wav, drop = FALSE]
delay <- delay[1:n_delay]
wavelength <- wavelength[1:n_wav]

compute_lof <- function(model, data) {
  100 * sqrt(sum((data - model)^2) / sum(data^2))
}

# SVD reference
sv <- svd(mat, nu = 10, nv = 10)
svd_lof <- c()
mod <- matrix(0, nrow = nrow(mat), ncol = ncol(mat))
for (i in 1:10) {
  mod <- mod + sv$d[i] * (sv$u[, i] %o% sv$v[, i])
  svd_lof[i] <- compute_lof(mod, mat)
}

# Simple ALS reference (rank=2, NNLS updates)
rank <- 2
C <- abs(sv$u[, 1:rank, drop = FALSE])
S <- abs(sv$v[, 1:rank, drop = FALSE])

for (iter in 1:80) {
  # Update C row-wise
  for (i in 1:nrow(mat)) {
    fit_c <- nnls::nnls(S, mat[i, ])
    C[i, ] <- fit_c$x
  }
  # Update S row-wise over wavelengths (solve S[j,])
  for (j in 1:ncol(mat)) {
    fit_s <- nnls::nnls(C, mat[, j])
    S[j, ] <- fit_s$x
  }
}

als_model <- C %*% t(S)
als_lof <- compute_lof(als_model, mat)

json_path <- file.path(fixture_dir, "parity_fixture_tmp.json")
jsonlite::write_json(
  list(
    matrix = unname(as.vector(t(mat))),
    matrix_nrow = nrow(mat),
    matrix_ncol = ncol(mat),
    delay = unname(as.vector(delay)),
    wavelength = unname(as.vector(wavelength)),
    svd_lof_first10 = unname(as.vector(svd_lof)),
    als_lof = as.numeric(als_lof),
    als_model = unname(as.vector(t(als_model))),
    als_model_nrow = nrow(als_model),
    als_model_ncol = ncol(als_model)
  ),
  json_path,
  pretty = FALSE,
  auto_unbox = TRUE
)

npz_path <- file.path(fixture_dir, "parity_fixture.npz")
py_cmd <- sprintf(
  "import json, numpy as np; " %+%
    "d=json.load(open(r'%s')); " %+%
    "m=np.array(d['matrix'],dtype=float).reshape((d['matrix_nrow'],d['matrix_ncol'])); " %+%
    "am=np.array(d['als_model'],dtype=float).reshape((d['als_model_nrow'],d['als_model_ncol'])); " %+%
    "np.savez(r'%s', matrix=m, delay=np.array(d['delay'],dtype=float), wavelength=np.array(d['wavelength'],dtype=float), svd_lof_first10=np.array(d['svd_lof_first10'],dtype=float), als_lof=float(d['als_lof']), als_model=am)",
  json_path,
  npz_path
)

status <- system2("python3", c("-c", py_cmd))
if (status != 0) {
  stop("Failed to convert JSON to NPZ via python3.")
}

unlink(json_path)
cat(sprintf("Generated fixture: %s\n", npz_path))
