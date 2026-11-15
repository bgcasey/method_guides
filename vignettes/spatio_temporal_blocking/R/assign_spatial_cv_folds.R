#' Choose CV folds with a minimum detection count
#'
#' Select the largest fold count so each fold can receive at
#' least `min_detections` rows where `class_column == 1`.
#' The computation respects block-level detection totals because
#' fold assignment in [assign_spatial_cv_folds()] cycles fold IDs
#' within each block.
#'
#' @param data Data frame that holds the records to be folded.
#' @param block_column Column name for spatial blocks.
#' @param class_column Column name for class labels (detections).
#' @param min_detections Minimum detection rows (value 1) per fold.
#'   Default 1.
#' @param max_folds Largest fold count to consider. Default 10.
#' @param min_folds Smallest fold count to accept. Default 2.
#'   If nothing in `[min_folds, max_folds]` works, the function
#'   falls back to 1 fold when possible.
#'
#' @return Integer fold count satisfying the detection constraint.
#'   Throws an error if no folds are possible.
#'
#' @example
#' det_data <- tibble::tibble(
#'   block = rep(letters[1:2], each = 5),
#'   y = c(1, 1, 1, 1, 0, 1, 1, 0, 0, 1)
#' )
#'
#' choose_folds_min_detections(
#'   data = det_data,
#'   block_column = "block",
#'   class_column = "y",
#'   min_detections = 1,
#'   max_folds = 5
#' )
choose_folds_min_detections <- function(
    data,
    block_column,
    class_column,
    min_detections = 1L,
    max_folds = 10L,
    min_folds = 2L
) {
  if (!is.numeric(min_detections) ||
      min_detections < 1) {
    stop("min_detections must be >= 1.")
  }

  det_counts <- data %>%
    dplyr::filter(.data[[class_column]] == 1) %>%
    dplyr::count(
      !!sym(block_column),
      name = "n_detect"
    )

  if (nrow(det_counts) == 0 ||
      sum(det_counts$n_detect) == 0) {
    stop("No detections found to allocate across folds.")
  }

  fold_cap <- min(max_folds, max(det_counts$n_detect))
  best <- NA_integer_

  check_folds <- function(k) {
    counts <- det_counts$n_detect
    vapply(
      seq_len(k),
      function(fold_idx) {
        contrib <- ifelse(
          counts >= fold_idx,
          floor((counts - fold_idx) / k) + 1L,
          0L
        )
        as.integer(sum(contrib))
      },
      integer(1)
    )
  }

  for (k in seq(
    from = fold_cap,
    to = min_folds,
    by = -1
  )) {
    per_fold <- check_folds(k)
    if (min(per_fold) >= min_detections) {
      best <- k
      break
    }
  }

  if (is.na(best)) {
    per_fold <- sum(det_counts$n_detect)
    if (per_fold >= min_detections) {
      best <- 1L
    } else {
      stop("Insufficient detections for even one fold.")
    }
  }

  return(best)
}

#'
#' Assign Spatial and Class-Stratified Cross-Validation Folds
#'
#' This function assigns cross-validation (CV) folds to a dataset.
#' The assignment is stratified by both a spatial block column and a
#' class column (e.g., detection status), ensuring balanced class
#' proportions within each spatial block across all folds.
#'
#' @param data A data frame containing the dataset to which CV folds
#'   will be assigned.
#' @param block_column A string specifying the column name for spatial
#'   blocks for stratification.
#' @param class_column A string specifying the column name for class
#'   stratification (e.g., detection status).
#' @param n_folds An integer specifying the number of folds to assign.
#'
#' @return A data frame with an additional column named `fold`, which
#'   indicates the assigned fold for each row.
#'
#' @example
#' my_data <- data.frame(
#'   id = 1:500,
#'   spatio_temporal_block = rep(1:10, each = 50),
#'   detection = sample(0:1, 500, replace = TRUE),
#'   value = round(runif(500, min = 10, max = 100), 2)
#' )
#' 
#' data_with_folds <- assign_spatial_cv_folds(
#'   data = my_data,
#'   block_column = "spatio_temporal_block",
#'   class_column = "detection",
#'   n_folds = 5
#' )
#' 
#' block_counts <- data_with_folds %>%
#'   group_by(fold, spatio_temporal_block, detection) %>%
#'   summarise(
#'     count = n(),
#'     .groups = "drop"
#'   ) %>%
#'   arrange(fold, spatio_temporal_block, detection)
#' 
#' print(block_counts)
assign_spatial_cv_folds <- function(
    data,
    block_column,
    class_column,
    n_folds
) {
  data %>%
    dplyr::group_by(
      !!sym(block_column),
      !!sym(class_column)
    ) %>%
    dplyr::group_split() %>%
    purrr::map_dfr(
      ~{
        n <- nrow(.x)
        fold_ids <- sample(
          rep(seq_len(n_folds), length.out = n)
        )
        .x %>%
          dplyr::mutate(fold = fold_ids)
      }
    )
}
