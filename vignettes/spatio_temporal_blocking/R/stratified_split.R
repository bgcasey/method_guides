#' Stratified Split for Training and Test Data with Class Balance
#'
#' This function performs stratified sampling by splitting 
#' the dataset into training and test sets based on a 
#' specified fraction for training. Stratification is performed 
#' using values in the specified block column and the class 
#' column, ensuring proportional representation of each group 
#' and class in both training and test sets.
#'
#' @param data A data frame containing the dataset to be split.
#' @param block_column A string specifying the column name to use 
#'   for stratification.
#' @param class_column A string specifying the column name for 
#'   class (e.g., detection status).
#' @param train_frac A numeric value between 0 and 1 indicating 
#'   the fraction of data to allocate to the training set.
#'
#' @return A data frame with an additional column `split` indicating 
#'   whether each row belongs to the "train" or "test" set.
#'
#' @example
#' # Example usage of the function
#' stratified_data <- stratified_split(
#'   data = my_data, 
#'   block_column = "group", 
#'   class_column = "detection",
#'   train_frac = 0.7
#' )
#' table(stratified_data$split, stratified_data$detection)
#'
stratified_split <- function(
    data,
    block_column,
    class_column,
    train_frac
) {
  data %>%
    # One group per block x class (e.g. detections in block_10)
    group_by(
      !!sym(block_column),
      !!sym(class_column)
    ) %>%
    group_split() %>%
    # Split each group separately, then bind the groups back together
    map_dfr(~ {
      n <- nrow(.x)
      # Rows to send to training; rounding means groups of 1-2 rows
      # always go entirely to training
      train_size <- round(train_frac * n)
      .x %>%
        mutate(
          # Randomly pick train_size rows for training; the rest are test
          split = ifelse(
            row_number() %in% sample(1:n, train_size),
            "train",
            "test"
          )
        )
    })
}
