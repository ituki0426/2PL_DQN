# Run from the repository: Rscript tests/test_estimate_irw_2pl.R
source('estimate_irw_2pl.R')
suppressPackageStartupMessages(library(mirt))

expect_error <- function(f, pattern) {
  msg <- tryCatch({ f(); NULL }, error = function(e) conditionMessage(e))
  stopifnot(!is.null(msg), grepl(pattern, msg, fixed = TRUE))
}

run_tests <- function() {
  tmp <- tempfile('irw-2pl-test-')
  dir.create(tmp)
  input <- file.path(tmp, 'input')
  dir.create(input)
  cfg <- parse_args(c('--input-dir', input, '--output-dir', file.path(tmp, 'output')))
  set.seed(135)
  a <- seq(0.8, 1.8, length.out = 12)
  b <- seq(-1.5, 1.5, length.out = 12)
  x <- mirt::simdata(matrix(a), -a * b, N = 3000, itemtype = '2PL')
  raw <- data.frame(id = sprintf('%05d', seq_len(nrow(x))), x, check.names = FALSE)
  names(raw)[-1L] <- c('001', 'NA', 'item space', 'x-y', paste0('q', 5:12))
  raw[1, 2] <- NA
  raw <- rbind(raw, c('all_missing', rep(NA, 12)))
  path <- file.path(input, 'synthetic_wide.csv')
  write.csv(raw, path, row.names = FALSE, na = '')
  before <- tools::md5sum(path)
  dat <- read_wide(path)
  stopifnot(dat$n_removed == 1, nrow(dat$responses) == 3000,
            dat$n_missing == 13, identical(dat$item_ids, names(raw)[-1L]),
            is.na(dat$responses[1, 1]), length(dat$respondent_ids) == 3001,
            identical(dat$respondent_ids[1], '00001'), !dat$keep[3001],
            dat$person_n_observed[3001] == 0, dat$person_n_correct[3001] == 0)
  stopifnot(fit_file(path, cfg), identical(before, tools::md5sum(path)))
  out <- file.path(cfg$output_dir, 'synthetic')
  result <- read.csv(file.path(out, 'item_parameters.csv'), colClasses = c(item = 'character'),
                     na.strings = NULL, check.names = FALSE)
  stopifnot(identical(result$item, names(raw)[-1L]), all(result$converged),
            max(abs(result$a - a)) < 0.4, max(abs(result$b - b)) < 0.4)
  scores <- read.csv(file.path(out, 'person_scores.csv'), colClasses = c(id = 'character'),
                     na.strings = 'NA', check.names = FALSE)
  stopifnot(nrow(scores) == nrow(raw), identical(scores$id, raw$id),
            all(is.finite(scores$theta_EAP[-nrow(scores)])),
            all(is.finite(scores$SE_EAP[-nrow(scores)])),
            all(scores$SE_EAP[-nrow(scores)] > 0),
            is.na(scores$theta_EAP[nrow(scores)]), is.na(scores$SE_EAP[nrow(scores)]),
            scores$n_observed[nrow(scores)] == 0, scores$n_correct[nrow(scores)] == 0,
            cor(scores$theta_EAP[-nrow(scores)], rowSums(x)) > 0.9)
  saved <- readRDS(file.path(out, 'model.rds'))
  raw_coef <- coef(saved$model, simplify = TRUE)
  stopifnot(all.equal(result$b, unname(-raw_coef$items[, 'd'] / raw_coef$items[, 'a1'])),
            unname(raw_coef$means) == 0, unname(raw_coef$cov) == 1,
            all(raw_coef$items[, 'g'] == 0), all(raw_coef$items[, 'u'] == 1))
  stopifnot(all(file.exists(file.path(out, c('person_scores.csv', 'fit_summary.csv',
                                            'warnings.txt', 'session_info.txt')))))

  # Identical files fit independently and receive separate output directories.
  file.copy(path, file.path(input, 'copy_wide.csv'))
  stopifnot(fit_file(file.path(input, 'copy_wide.csv'), cfg))
  copy <- read.csv(file.path(cfg$output_dir, 'copy', 'item_parameters.csv'),
                   colClasses = c(item = 'character'), na.strings = NULL)
  stopifnot(identical(result, copy))

  short_cfg <- cfg
  short_cfg$output_dir <- file.path(tmp, 'nonconverged')
  short_cfg$max_cycles <- 1L
  stopifnot(!fit_file(path, short_cfg))
  status <- read.csv(file.path(short_cfg$output_dir, 'synthetic', 'fit_summary.csv'))
  stopifnot(!status$converged)

  bad <- raw
  bad[2, 2] <- '2'
  bad_path <- file.path(input, 'bad_wide.csv')
  write.csv(bad, bad_path, row.names = FALSE, na = '')
  expect_error(function() read_wide(bad_path), 'Responses must be')
  bad <- raw
  bad[[2]] <- 1
  write.csv(bad, bad_path, row.names = FALSE, na = '')
  expect_error(function() read_wide(bad_path), 'without both 0 and 1')
  bad <- raw
  bad$id[2] <- bad$id[1]
  write.csv(bad, bad_path, row.names = FALSE, na = '')
  expect_error(function() read_wide(bad_path), 'Respondent IDs')
  expect_error(function() parse_args(c('--quadpts', '2')), 'Invalid integer')
  expect_error(function() parse_args(c('--tol', '0')), 'tol must be positive')
  expect_error(function() main(c('--input-dir', input, '--file', 'missing_wide.csv')), 'No matching')
  cat('PASS: MML recovery, EAP person scores, parameter convention, independent fits,\n',
      'missing data, original IDs, unchanged inputs, non-convergence and invalid-input handling.\n',
      'Temporary test output:', tmp, '\n')
}

run_tests()
