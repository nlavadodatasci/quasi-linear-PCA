# Ground truth from the R package Gifi (princals) for comparison/princals/compare.py.
#
# Reads manifest.json and the input CSVs exported by `python compare.py inputs`
# (results/inputs/, not stored in the repository), runs Gifi::princals with
#   - the explicit knots of the manifest (degrees, ordinal and eps per run), and
#   - the package defaults (run "defaults"),
# and writes one JSON file per data set to results/r_ground_truth/.
#
# Run from comparison/princals/:   Rscript generate_r_ground_truth.R
# Requires: Gifi (1.0-0), jsonlite.  Gifi is GPL-3; it is only called here.
#
# JSON layout (per data set):
#   meta: R and Gifi versions, input md5, date, (neumann) agreement with data(neumann)
#   runs: {run: {evals, objectscores (n x ndim), transform (n x m), loadings (ndim x m; compare.py transposes),
#                ntel, f, params (rank of the centred B-spline basis per variable)}}

suppressPackageStartupMessages({
  library(Gifi)
  library(jsonlite)
})

manifest <- fromJSON("manifest.json", simplifyVector = FALSE)
ndim <- manifest$ndim
outdir <- file.path("results", "r_ground_truth")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

if (packageVersion("Gifi") != package_version(manifest$packages$Gifi))   # "1.0-0" == "1.0.0"
  warning("Gifi version ", packageVersion("Gifi"), " differs from the manifest (", manifest$packages$Gifi, ")")

runs_for <- function(name) {
  Filter(function(run) {
    ds <- manifest$runs[[run]]$datasets
    is.null(ds) || name %in% unlist(ds)
  }, names(manifest$runs))
}

basis_rank <- function(x, degree, knots) {
  b <- Gifi:::bsplineBasis(x, degree, knots)
  b <- b[, colSums(b) > 0, drop = FALSE]
  qr(scale(b, center = TRUE, scale = FALSE))$rank
}

for (name in names(manifest$datasets)) {
  input <- manifest$inputs[[name]]
  path <- input$file
  if (!file.exists(path)) stop("missing input ", path, ": run `python compare.py inputs` first")
  md5 <- unname(tools::md5sum(path))
  if (md5 != input$md5) stop("input ", path, " does not match the manifest (md5 ", md5, ")")
  df <- read.csv(path, check.names = FALSE)
  stopifnot(identical(colnames(df), unlist(input$columns)))

  meta <- list(dataset = name, input_md5 = md5, R = R.version.string,
               Gifi = as.character(packageVersion("Gifi")),
               jsonlite = as.character(packageVersion("jsonlite")),
               date = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"))
  if (name == "neumann") {
    e <- new.env()
    data("neumann", package = "Gifi", envir = e)
    g <- e$neumann
    meta$neumann_matches_gifi_data <- isTRUE(all.equal(unname(as.matrix(df)),
                                                       unname(as.matrix(g[, colnames(df)])),
                                                       check.attributes = FALSE))
  }

  runs <- list()
  for (run in runs_for(name)) {
    spec <- manifest$runs[[run]]
    if (isTRUE(spec$package_defaults)) {
      fit <- princals(df, ndim = ndim)
      knots_used <- fit$knots
      deg <- rep(1, ncol(df))
    } else {
      kn <- manifest$knots[[name]][[as.character(spec$n_knots)]]
      knots_used <- lapply(colnames(df), function(col) as.numeric(unlist(kn[[col]])))
      deg <- rep(spec$degree, ncol(df))
      eps <- if (is.null(spec$eps)) manifest$fit$eps else spec$eps
      fit <- princals(df, ndim = ndim, knots = knots_used, degrees = spec$degree,
                      ordinal = rep(spec$ordinal, ncol(df)), eps = eps, itmax = manifest$fit$itmax)
    }
    params <- tryCatch(vapply(seq_len(ncol(df)), function(j)
      basis_rank(df[[j]], deg[j], knots_used[[j]]), numeric(1)), error = function(e) numeric(0))
    runs[[run]] <- list(
      evals = unname(fit$evals),
      objectscores = unname(as.matrix(fit$objectscores)),
      transform = unname(as.matrix(fit$transform)),
      loadings = t(unname(as.matrix(fit$loadings))),
      ntel = fit$ntel,
      f = fit$f,
      params = params)
    cat(sprintf("%-12s %-14s evals %s  ntel %d\n", name, run,
                paste(format(fit$evals[1:ndim], digits = 6), collapse = " "), fit$ntel))
  }
  write_json(list(meta = meta, runs = runs), file.path(outdir, paste0(name, ".json")),
             auto_unbox = TRUE, digits = I(17), matrix = "rowmajor", pretty = FALSE)
}
cat("wrote", outdir, "\n")
