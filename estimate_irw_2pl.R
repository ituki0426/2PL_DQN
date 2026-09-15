#!/usr/bin/env Rscript
# Independently calibrate each IRW wide CSV and estimate EAP person scores.

script_directory <- function() {
  arg <- grep('^--file=', commandArgs(FALSE), value = TRUE)
  if (length(arg)) dirname(normalizePath(sub('^--file=', '', arg[1]))) else getwd()
}

usage <- function() {
  cat('Usage: Rscript estimate_irw_2pl.R [options]\n',
      '  --input-dir DIR   Input directory (default: <script>/irw_datasets)\n',
      '  --output-dir DIR  Output directory (default: <script>/result/IRW_2PL)\n',
      '  --file NAME       Process one *_wide.csv in the input directory\n',
      '  --max-cycles N    EM iteration limit (default: 1000)\n',
      '  --quadpts N       Quadrature points (default: 61)\n',
      '  --tol X           EM convergence tolerance (default: 0.0001)\n',
      '  --seed N          Seed reset for each independent fit (default: 20260910)\n',
      '  --help            Show this help\n', sep = '')
}

parse_args <- function(args, root = script_directory()) {
  cfg <- list(input_dir = file.path(root, 'irw_datasets'),
              output_dir = file.path(root, 'result', 'IRW_2PL'), file = NULL,
              max_cycles = 1000L, quadpts = 61L, tol = 1e-4, seed = 20260910L)
  i <- 1L
  while (i <= length(args)) {
    flag <- args[i]
    if (flag == '--help') { usage(); return(NULL) }
    key <- gsub('-', '_', sub('^--', '', flag))
    if (!startsWith(flag, '--') || !key %in% names(cfg)) stop('Unknown option: ', flag)
    if (i == length(args)) stop('Missing value for ', flag)
    cfg[[key]] <- args[i + 1L]
    i <- i + 2L
  }
  for (key in c('max_cycles', 'quadpts', 'tol', 'seed')) {
    value <- suppressWarnings(as.numeric(cfg[[key]]))
    if (length(value) != 1L || !is.finite(value)) stop('Invalid value for ', key)
    if (key == 'tol') {
      if (value <= 0) stop('tol must be positive')
    } else {
      minimum <- switch(key, quadpts = 3, seed = 0, 1)
      if (value < minimum || value > .Machine$integer.max || value != floor(value))
        stop('Invalid integer for ', key)
      value <- as.integer(value)
    }
    cfg[[key]] <- value
  }
  cfg
}

read_wide <- function(path) {
  raw <- read.csv(path, colClasses = 'character', check.names = FALSE,
                  na.strings = NULL, fileEncoding = 'UTF-8-BOM', fill = FALSE)
  if (!nrow(raw) || !ncol(raw) || names(raw)[1] != 'id')
    stop('Expected a nonempty wide CSV with id as its first column')
  if (any(!nzchar(trimws(names(raw)))) || anyDuplicated(names(raw)))
    stop('Column names must be nonempty and unique')
  if (any(!nzchar(trimws(raw$id))) || anyDuplicated(raw$id))
    stop('Respondent IDs must be nonempty and unique')
  if (ncol(raw) < 4L) stop('At least three item columns are required')
  item_ids <- names(raw)[-1L]
  values <- as.matrix(raw[-1L])
  values[] <- trimws(values)
  missing <- values %in% c('', 'NA')
  invalid <- !(values %in% c('0', '1', '', 'NA'))
  if (any(invalid)) stop('Responses must be 0, 1, blank or NA; found: ',
                         paste(head(unique(values[invalid]), 5L), collapse = ', '))
  values[missing] <- NA_character_
  responses <- matrix(as.integer(values), nrow = nrow(raw), ncol = length(item_ids))
  n_observed <- colSums(!is.na(responses))
  n_correct <- colSums(responses, na.rm = TRUE)
  invalid_items <- n_observed == 0L | n_correct == 0L | n_correct == n_observed
  if (any(invalid_items)) stop('Items without both 0 and 1 cannot be calibrated: ',
                               paste(item_ids[invalid_items], collapse = ', '))
  keep <- rowSums(!is.na(responses)) > 0L
  if (sum(keep) < 2L) stop('At least two respondents with observed responses are required')
  person_n_observed <- rowSums(!is.na(responses))
  person_n_correct <- rowSums(responses, na.rm = TRUE)
  # Use safe internal names; retain the original item IDs in exported parameters.
  colnames(responses) <- sprintf('item_%06d', seq_along(item_ids))
  list(responses = responses[keep, , drop = FALSE], item_ids = item_ids,
       respondent_ids = raw$id, keep = keep, person_n_observed = person_n_observed,
       person_n_correct = person_n_correct,
       n_input = nrow(raw), n_removed = sum(!keep), n_missing = sum(is.na(responses)),
       n_observed = n_observed, n_correct = n_correct)
}

fit_file <- function(path, cfg) {
  dat <- read_wide(path)
  set.seed(cfg$seed)
  warnings <- character()
  started <- proc.time()[['elapsed']]
  fit <- withCallingHandlers(
    mirt::mirt(dat$responses, model = 1, itemtype = '2PL', method = 'EM',
               dentype = 'Gaussian', SE = FALSE, quadpts = cfg$quadpts,
               TOL = cfg$tol, technical = list(NCYCLES = cfg$max_cycles), verbose = FALSE),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart('muffleWarning')
    })
  elapsed <- proc.time()[['elapsed']] - started
  coefficients <- mirt::coef(fit, IRTpars = TRUE, simplify = TRUE)$items
  converged <- isTRUE(mirt::extract.mirt(fit, 'converged'))
  finite_parameters <- all(is.finite(coefficients[, c('a', 'b')]))
  eap <- withCallingHandlers(
    mirt::fscores(fit, method = 'EAP', response.pattern = dat$responses,
                  full.scores = TRUE, full.scores.SE = TRUE,
                  quadpts = cfg$quadpts, verbose = FALSE),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart('muffleWarning')
    }
  )
  theta <- rep(NA_real_, dat$n_input)
  theta_se <- rep(NA_real_, dat$n_input)
  theta[dat$keep] <- eap[, 'F1']
  theta_se[dat$keep] <- eap[, 'SE_F1']
  finite_scores <- all(is.finite(theta[dat$keep])) && all(is.finite(theta_se[dat$keep]))
  if (!converged) warnings <- c(warnings, 'EM did not converge; estimates are provisional.')
  if (!finite_parameters) warnings <- c(warnings, 'Non-finite item parameters were produced.')
  if (!finite_scores) warnings <- c(warnings, 'Non-finite EAP person scores were produced.')
  if (any(coefficients[, 'a'] <= 0))
    warnings <- c(warnings, 'Non-positive discrimination found; check item coding and model fit.')
  parameters <- data.frame(item = dat$item_ids, a = coefficients[, 'a'],
                           b = coefficients[, 'b'], n_observed = dat$n_observed,
                           n_correct = dat$n_correct, converged = converged,
                           row.names = NULL, check.names = FALSE)
  person_scores <- data.frame(id = dat$respondent_ids, theta_EAP = theta, SE_EAP = theta_se,
                              n_observed = dat$person_n_observed,
                              n_correct = dat$person_n_correct,
                              row.names = NULL, check.names = FALSE)
  summary <- data.frame(
    input_file = normalizePath(path), model = 'unidimensional_2PL', method = 'MML_EM',
    theta_mean = 0, theta_variance = 1, score_method = 'EAP', logistic_D = 1,
    n_input = dat$n_input, n_used = nrow(dat$responses),
    n_all_missing_removed = dat$n_removed, n_items = length(dat$item_ids),
    n_missing_input = dat$n_missing, converged = converged,
    finite_parameters = finite_parameters, finite_scores = finite_scores,
    iterations = mirt::extract.mirt(fit, 'iterations'),
    logLik = mirt::extract.mirt(fit, 'logLik'),
    AIC = mirt::extract.mirt(fit, 'AIC'), BIC = mirt::extract.mirt(fit, 'BIC'),
    quadpts = cfg$quadpts, max_cycles = cfg$max_cycles, tol = cfg$tol, seed = cfg$seed,
    elapsed_seconds = elapsed, R_version = as.character(getRversion()),
    mirt_version = as.character(utils::packageVersion('mirt')))
  out <- file.path(cfg$output_dir, sub('_wide\\.csv$', '', basename(path)))
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  write.csv(parameters, file.path(out, 'item_parameters.csv'), row.names = FALSE, na = 'NA')
  write.csv(person_scores, file.path(out, 'person_scores.csv'), row.names = FALSE, na = 'NA')
  write.csv(summary, file.path(out, 'fit_summary.csv'), row.names = FALSE, na = 'NA')
  # Preserve the fit for later diagnostics and the map from internal names to item IDs.
  saveRDS(list(model = fit, item_ids = dat$item_ids, summary = summary), file.path(out, 'model.rds'))
  writeLines(unique(warnings), file.path(out, 'warnings.txt'))
  writeLines(capture.output(utils::sessionInfo()), file.path(out, 'session_info.txt'))
  message(basename(path), ': ', nrow(dat$responses), ' respondents x ', length(dat$item_ids),
          ' items; converged=', converged, '; ', round(elapsed, 1), ' sec; output: ', out)
  for (w in unique(warnings)) message('  Warning: ', w)
  converged && finite_parameters && finite_scores
}

main <- function(args = commandArgs(TRUE)) {
  cfg <- parse_args(args)
  if (is.null(cfg)) return(invisible(TRUE))
  if (!requireNamespace('mirt', quietly = TRUE))
    stop('Install mirt first: Rscript -e \'install.packages("mirt", repos="https://cloud.r-project.org")\'')
  if (!dir.exists(cfg$input_dir)) stop('Input directory does not exist: ', cfg$input_dir)
  files <- sort(list.files(cfg$input_dir, pattern = '_wide\\.csv$', full.names = TRUE))
  if (!is.null(cfg$file)) files <- files[basename(files) == cfg$file]
  if (!length(files)) stop('No matching *_wide.csv files found')
  ok <- vapply(files, function(path) {
    tryCatch(fit_file(path, cfg), error = function(e) {
      message(basename(path), ': ERROR: ', conditionMessage(e))
      FALSE
    })
  }, logical(1))
  if (!all(ok)) stop(sum(!ok), ' dataset(s) failed or did not converge; see messages and fit_summary.csv')
  invisible(TRUE)
}

if (sys.nframe() == 0L) {
  tryCatch(main(), error = function(e) {
    message('Error: ', conditionMessage(e))
    quit(status = 1L)
  })
}
