library(dplyr)
library(purrr)
library(dggridR)

source("R/sample_blocks_binomial.R")
source("R/stratified_split.R")
source("R/assign_spatial_cv_folds.R")
source("R/assign_bootstrap_ids_matrix.R")

# Simulate 3,000 survey locations across Alberta, 2000-2024.
# Detection probability rises toward the south-east so that
# some blocks have many detections and others have none.
set.seed(42)
n <- 3000
dummy <- tibble(
  longitude = runif(n, -120, -110),
  latitude  = runif(n, 49, 60),
  year      = sample(2000:2024, n, replace = TRUE)
) %>%
  mutate(
    p_det    = plogis(-3.5 + 0.35 * (60 - latitude) / 2 +
                        0.2 * (longitude + 120) / 2),
    presence = rbinom(n, 1, p_det)
  ) %>%
  select(-p_det)

## Step 1: spatio-temporal blocks (same logic as 03_02 section 2)
grid <- dgconstruct(spacing = 150, metric = TRUE)

dummy <- dummy %>%
  mutate(
    spatial_block = dgGEO_to_SEQNUM(grid, longitude, latitude)$seqnum,
    temporal_block = cut(
      year,
      breaks = quantile(year, probs = seq(0, 1, by = 0.2), na.rm = TRUE),
      include.lowest = TRUE,
      right = FALSE
    ),
    spatio_temporal_block = interaction(
      droplevels(as.factor(spatial_block)),
      droplevels(as.factor(temporal_block)),
      sep = "::",
      drop = TRUE
    )
  ) %>%
  group_by(spatio_temporal_block) %>%
  mutate(spatio_temporal_block = paste0("block_", cur_group_id())) %>%
  ungroup()

cat("\n== Step 1: block summary ==\n")
dummy %>%
  summarise(
    n_obs = n(),
    n_detections = sum(presence),
    n_spatial_blocks = n_distinct(spatial_block),
    n_spatio_temporal_blocks = n_distinct(spatio_temporal_block)
  ) %>%
  print()
print(table(dummy$temporal_block))

## Step 2: subsample absences within blocks
model_data <- dummy %>%
  select(presence, spatio_temporal_block, longitude, latitude, year)

sampled <- sample_blocks_binomial(
  model_data,
  block_column = "spatio_temporal_block",
  response_column = "presence",
  ratio = 10,
  min_absences = 2
)

cat("\n== Step 2: before and after absence subsampling ==\n")
bind_rows(
  model_data %>% mutate(stage = "before"),
  sampled %>% mutate(stage = "after")
) %>%
  group_by(stage) %>%
  summarise(
    n_blocks = n_distinct(spatio_temporal_block),
    detections = sum(presence == 1),
    absences = sum(presence == 0),
    .groups = "drop"
  ) %>%
  arrange(desc(stage)) %>%
  print()

## Step 3: training-test split
split_data <- stratified_split(
  data = sampled,
  block_column = "spatio_temporal_block",
  class_column = "presence",
  train_frac = 0.75
)
cat("\n== Step 3: split by class ==\n")
print(table(split = split_data$split, presence = split_data$presence))

train_data <- split_data %>% filter(split == "train") %>% select(-split)

## Step 4: choose folds and assign them
folds <- choose_folds_min_detections(
  data = train_data,
  block_column = "spatio_temporal_block",
  class_column = "presence",
  min_detections = 25,
  max_folds = 5
)
cat("\n== Step 4: folds chosen ==\n")
print(folds)

train_data <- assign_spatial_cv_folds(
  data = train_data,
  block_column = "spatio_temporal_block",
  class_column = "presence",
  n_folds = folds
)

cat("\n== Step 4: detections and absences per fold ==\n")
train_data %>%
  group_by(fold) %>%
  summarise(
    detections = sum(presence == 1),
    absences = sum(presence == 0),
    n_blocks = n_distinct(spatio_temporal_block),
    .groups = "drop"
  ) %>%
  print()

cat("\n== Step 4: folds represented per block ==\n")
train_data %>%
  group_by(spatio_temporal_block) %>%
  summarise(n_folds_in_block = n_distinct(fold), n = n(), .groups = "drop") %>%
  filter(n >= folds) %>%
  count(n_folds_in_block, name = "n_blocks") %>%
  print()

## Step 6: stratified bootstrap
bootstrap_ids <- assign_bootstrap_ids_m(
  data = train_data,
  block_column = "spatio_temporal_block",
  class_column = "presence",
  num_boot = 50,
  max_sample_size = nrow(train_data),
  seed = 123,
  replace = TRUE,
  progress = FALSE
)
cat("\n== Step 6: bootstrap index matrix ==\n")
print(dim(bootstrap_ids))
cat("Unique rows in replicate 1:",
    length(unique(bootstrap_ids[, 1])), "of", nrow(train_data), "\n")
cat("Replicates identical to replicate 1:",
    sum(apply(bootstrap_ids, 2, identical, bootstrap_ids[, 1])), "of 50\n")
