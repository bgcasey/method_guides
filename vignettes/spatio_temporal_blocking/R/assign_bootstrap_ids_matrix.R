#' Assign Stratified Bootstrap Indices (Block + Class, Dynamic Proportion)
#'
#' Creates a matrix of stratified bootstrap sample indices using both
#' block and class columns. The proportion of rows sampled in each
#' replicate is dynamically determined so that the total sample size
#' does not exceed `max_sample_size`. Each bootstrap will have the 
# same relative proportion of classes as the original data.
#'
#' @param data data.frame. Dataset containing block and class columns.
#' @param block_column character. Name of the block column.
#' @param class_column character. Name of the class column (e.g. 0/1).
#' @param num_boot integer. Number of bootstrap replicates (columns).
#' @param max_sample_size integer. Maximum number of rows sampled in
#'   each bootstrap replicate.
#' @param replace logical. Sample with replacement? Default = TRUE.
#' @param seed integer or NULL. Optional random seed.
#' @param progress logical. If TRUE, show a progress bar.
#'
#' @return Integer matrix (#sampled_rows x num_boot). Columns named
#'   boot_1 ... boot_{num_boot}. Entry (i, j) is the original row index
#'   at position i in replicate j.
#'
#' @example
#' set.seed(1)
#' n_blocks <- 10
#' n_per_block <- 2000
#' block_ids <- sprintf("BLK_%02d", seq_len(n_blocks))
#' data <- data.frame(
#'   block = rep(block_ids, each = n_per_block),
#'   class = sample(0:1, n_blocks * n_per_block, TRUE)
#' )
#' idx_mat <- assign_bootstrap_ids_m(
#'   data = data,
#'   block_column = "block",
#'   class_column = "class",
#'   num_boot = 5,
#'   max_sample_size = 10000,
#'   replace = TRUE,
#'   seed = 42
#' )
#' dim(idx_mat)
#' head(idx_mat[, 1])
assign_bootstrap_ids_m <- function(
    data,
    block_column,
    class_column,
    num_boot = 100,
    max_sample_size = 10000,
    replace = TRUE,
    seed = NULL,
    progress = TRUE
) {
  # Step 1: Validate inputs ----
  if (!is.data.frame(data)) {
    stop("data must be a data.frame.")
  }
  if (!block_column %in% names(data)) {
    stop("block_column not found in data.")
  }
  if (!class_column %in% names(data)) {
    stop("class_column not found in data.")
  }
  if (!is.null(seed)) set.seed(seed)
  n <- nrow(data)
  if (n == 0L) stop("data has zero rows.")
  if (max_sample_size <= 0 || max_sample_size > n) {
    stop("max_sample_size must be > 0 and <= number of rows.")
  }

  # Step 2: Split indices by block and class ----
  # group_indices is a list of row numbers, one element per block x class
  block_vec <- data[[block_column]]
  class_vec <- data[[class_column]]
  if (is.factor(block_vec)) block_vec <- droplevels(block_vec)
  if (is.factor(class_vec)) class_vec <- droplevels(class_vec)
  row_ids <- seq_len(n)
  group_indices <- split(
    row_ids,
    list(block = block_vec, class = class_vec),
    drop = TRUE
  )

  group_sizes <- vapply(group_indices, length, integer(1))
  total_group_size <- sum(group_sizes)

  # Step 3: Calculate dynamic proportion ----
  # prop = 1 when max_sample_size equals the number of rows
  prop <- min(1, max_sample_size / total_group_size)

  # Step 4: Determine sample sizes ----
  # Rows to draw from each group (at least 1)
  sample_sizes <- pmax(1, floor(group_sizes * prop))
  total_sample_size <- sum(sample_sizes)

  # Step 5: Allocate result matrix ----
  # One row per sampled position, one column per replicate
  idx_mat <- matrix(
    NA_integer_,
    nrow = total_sample_size,
    ncol = num_boot
  )
  colnames(idx_mat) <- paste0("boot_", seq_len(num_boot))

  # Step 6: Define sampler for one replicate ----
  # Resample row numbers within each group, then concatenate the groups
  sample_one <- function() {
    unlist(
      mapply(function(ix, m) {
        ix[sample.int(
          n = length(ix),
          size = m,
          replace = replace
        )]
      }, group_indices, sample_sizes, SIMPLIFY = FALSE),
      use.names = FALSE
    )
  }

  # Step 7: Generate all replicates ----
  if (progress) {
    pb <- utils::txtProgressBar(min = 0,
                                max = num_boot,
                                style = 3)
  }
  for (b in seq_len(num_boot)) {
    idx_mat[, b] <- sample_one()
    if (progress) utils::setTxtProgressBar(pb, b)
  }
  if (progress) close(pb)

  return(idx_mat)
}
