#' Stratified Sampling for Binomial Response Within Blocks
#'
#' This function samples rows within each block for a binomial
#' response variable. All detections (response == 1) are included
#' and a specified ratio of absences (response == 0) are randomly
#' sampled without replacement. If the total sample size exceeds
#' max_sample_size, the function downsamples both detections and
#' absences *proportionally* to preserve the overall ratio and
#' block-level stratification as much as possible.
#'
#' @param data Data frame to sample from
#' @param block_column Character. Block column name
#' @param response_column Character. Binomial response column
#' @param ratio Numeric. Ratio of absences to detections
#' @param min_absences Integer. Minimum absences per block
#' @param max_sample_size Optional integer. Maximum total sample size
#' @return Data frame of sampled rows
#' 
#' @example
#' set.seed(42)
#' df <- data.frame(
#'   block_id = rep(letters[1:3], each = 10),
#'   species = c(rep(1, 2), rep(0, 8), 
#'               rep(1, 1), rep(0, 9),
#'               rep(0, 10)),
#'   val = rnorm(30)
#' )
#' df_out <- sample_blocks_binomial(
#'   data = df,
#'   block_column = "block_id",
#'   response_column = "species",
#'   ratio = 3,
#'   min_absences = 5,
#'   max_sample_size = 12
#' )
#' dplyr::count(df_out, block_id, species)
sample_blocks_binomial <- function(
    data,
    block_column,
    response_column,
    ratio = 5,
    min_absences = 10,
    max_sample_size = NULL
) {
  # Step 1: Check columns
  if (!block_column %in% names(data)) {
    stop(glue::glue(
      "Column '{block_column}' not found in data."
    ))
  }
  if (!response_column %in% names(data)) {
    stop(glue::glue(
      "Column '{response_column}' not found in data."
    ))
  }
  
  # Step 2: Filter blocks with detections only
  det_blocks <- data %>%
    dplyr::group_by(.data[[block_column]]) %>%
    dplyr::summarise(
      n_det = sum(.data[[response_column]] == 1),
      .groups = "drop"
    ) %>%
    dplyr::filter(n_det > 0) %>%
    dplyr::pull(.data[[block_column]])
  
  data <- data %>%
    dplyr::filter(.data[[block_column]] %in% det_blocks)
  
  # Step 3: Split by block (vectorized)
  blocks <- data %>%
    dplyr::group_split(.data[[block_column]])
  
  # Step 4: Stratified sampling per block
  sampled <- lapply(blocks, function(.x) {
    det <- .x[.x[[response_column]] == 1, , drop = FALSE]
    ps_abs <- .x[.x[[response_column]] == 0, , drop = FALSE]
    n_det <- nrow(det)
    n_ps_abs <- nrow(ps_abs)
    n_sample_abs <- min(n_ps_abs, max(ratio * n_det, min_absences))
    if (n_sample_abs == 0) return(det)
    ps_abs_sample <- ps_abs[sample(n_ps_abs, n_sample_abs), , drop = FALSE]
    rbind(det, ps_abs_sample)
  })
  
  # Step 5: Combine results
  result <- dplyr::bind_rows(sampled)
  
  # Step 6: If max_sample_size specified, downsample proportionally
  if (!is.null(max_sample_size) && nrow(result) > max_sample_size) {
    n_det <- sum(result[[response_column]] == 1)
    n_abs <- sum(result[[response_column]] == 0)
    prop_det <- n_det / nrow(result)
    prop_abs <- n_abs / nrow(result)
    n_det_final <- round(max_sample_size * prop_det)
    n_abs_final <- max_sample_size - n_det_final
    det_rows <- result[result[[response_column]] == 1, , drop = FALSE]
    abs_rows <- result[result[[response_column]] == 0, , drop = FALSE]
    # Downsample both classes (without replacement)
    if (nrow(det_rows) > n_det_final) {
      det_rows <- det_rows[sample(
        nrow(det_rows), n_det_final
      ), , drop = FALSE]
    }
    if (nrow(abs_rows) > n_abs_final) {
      abs_rows <- abs_rows[sample(
        nrow(abs_rows), n_abs_final
      ), , drop = FALSE]
    }
    result <- dplyr::bind_rows(det_rows, abs_rows)
  }
  
  return(result)
}
