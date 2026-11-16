# Spatio-temporal stratification of survey locations: worked example
# Run from this vignette's folder so the relative source() paths resolve.

# 0. Simulate survey data ----
library(dplyr)          # data manipulation
library(purrr)          # iteration (map_dfr, map_dbl)
library(ggplot2)        # block maps
library(sf)             # spatial vector data
library(dggridR)        # global hexagonal grid
library(rnaturalearth)  # provincial boundaries
library(MetBrewer)      # map colour palette
library(utilR) # theme_science_map()

# Helper functions (shown in full in the guide)
source("R/stratified_split.R")
source("R/assign_spatial_cv_folds.R")
source("R/assign_bootstrap_ids_matrix.R")

# Alberta boundary (WGS84), used to place the simulated survey locations
alberta <- ne_states(country = "Canada", returnclass = "sf") %>%
  filter(name_en == "Alberta")

set.seed(42)
n <- 10000

# Draw n random points inside Alberta and extract their coordinates
locations <- st_sample(alberta, size = n) %>%
  st_coordinates()

# One row per survey: location, a random survey year, and a 0/1 detection.
# Detection probability increases toward the south (lower latitude) and
# east (higher longitude), so detections cluster in some blocks.
data <- tibble(
  longitude = locations[, "X"],
  latitude  = locations[, "Y"],
  year      = sample(2000:2024, n, replace = TRUE)
) %>%
  mutate(
    p_det    = plogis(-3.5 + 0.35 * (60 - latitude) / 2 +
                        0.2 * (longitude + 120) / 2),
    presence = rbinom(n, 1, p_det)  # 1 = detected, 0 = not detected
  ) %>%
  select(-p_det)

# 1. Build spatial and temporal blocks ----
cat("
== Step 1: block summary ==
")
# Global hexagonal grid; dggridR picks the resolution whose cell-centre
# spacing is closest to 150 km
grid <- dgconstruct(
  spacing = 150,
  metric = TRUE
)

data <- data %>%
  mutate(
    # Assign points to spatial blocks using the precomputed grid
    spatial_block = dgGEO_to_SEQNUM(
      grid,
      longitude,
      latitude
    )$seqnum,
    # Create quantile-based temporal blocks
    # (breaks at the 20th, 40th, 60th, and 80th percentiles of year)
    temporal_block = cut(
      year,
      breaks = quantile(
        year,
        probs = seq(0, 1, by = 0.2),
        na.rm = TRUE
      ),
      include.lowest = TRUE,
      right = FALSE
    ),
    # Combine spatial and temporal blocks
    # (one level per observed cell x year-bin combination)
    spatio_temporal_block = interaction(
      droplevels(as.factor(spatial_block)),
      droplevels(as.factor(temporal_block)),
      sep = "::",
      drop = TRUE
    )
  ) %>%
  # Replace the long "cell::bin" labels with short IDs (block_1, block_2, ...)
  group_by(spatio_temporal_block) %>%
  mutate(
    spatio_temporal_block = paste0(
      "block_", cur_group_id()
    )
  ) %>%
  ungroup()

# Check: number of observations, detections, and blocks
data %>%
  summarise(
    n_obs = n(),
    n_detections = sum(presence),
    n_spatial_blocks = n_distinct(spatial_block),
    n_spatio_temporal_blocks = n_distinct(spatio_temporal_block)
  ) %>%
  print()
# Check: observations per year bin
print(table(data$temporal_block))

# Map the blocks
dir.create("figures", showWarnings = FALSE)

# Project the boundary to Canada Atlas Lambert (EPSG:3978) for mapping
alberta_lcc <- st_transform(alberta, crs = 3978)

# Convert the occupied grid cells to polygons and project them
block_cells <- dgcellstogrid(grid, cells = unique(data$spatial_block)) %>%
  st_as_sf() %>%
  mutate(seqnum = as.integer(seqnum)) %>%
  st_transform(crs = 3978)

# Locations per spatial block
spatial_counts <- block_cells %>%
  left_join(count(data, spatial_block, name = "n_obs"),
            by = c("seqnum" = "spatial_block"))

# Figure 1: spatial blocks
p1 <- ggplot(spatial_counts) +
  geom_sf(aes(fill = n_obs), colour = "white", linewidth = 0.3) +
  geom_sf(data = alberta_lcc, fill = NA, colour = "black", linewidth = 0.5) +
  scale_fill_met_c("Hokusai3", direction = -1) +
  labs(x = NULL, y = NULL, fill = "Locations per block") +
  theme_science_map()
ggsave("figures/spatial_blocks.png", p1, width = 6, height = 6, dpi = 150)

# Locations per spatial block within each year bin (one row per
# spatio-temporal block)
st_counts <- block_cells %>%
  inner_join(count(data, spatial_block, temporal_block, name = "n_obs"),
             by = c("seqnum" = "spatial_block"))

# Figure 2: spatio-temporal blocks, one panel per year bin
p2 <- ggplot(st_counts) +
  geom_sf(aes(fill = n_obs), colour = "white", linewidth = 0.2) +
  geom_sf(data = alberta_lcc, fill = NA, colour = "black", linewidth = 0.4) +
  facet_wrap(~temporal_block, nrow = 2) +
  scale_fill_met_c("Hokusai3", direction = -1) +
  scale_x_continuous(breaks = c(-120, -115, -110)) +
  labs(x = NULL, y = NULL, fill = "Locations per block") +
  theme_science_map() +
  theme(
    strip.text = element_text(face = "bold"),
    legend.position = "bottom",
    legend.key.width = unit(1.5, "cm")
  )
ggsave("figures/spatio_temporal_blocks.png", p2, width = 9, height = 8, dpi = 150)

# 2. Split training and test data by block ----
cat("
== Step 2: split by class ==
")
# Split 75/25 within each block x class group
split_data <- stratified_split(
  data = data,
  block_column = "spatio_temporal_block",
  class_column = "presence",
  train_frac = 0.75
)
# Check: rows per block, class, and split
count(split_data, spatio_temporal_block, presence, split)
# Check: overall split by class
table(split_data$split, split_data$presence)

# Keep the training rows for fold assignment and bootstrapping
train_data <- split_data %>% filter(split == "train") %>% select(-split)

# 3. Choose the number of folds and assign them ----
cat("
== Step 3: folds ==
")
# Largest fold count (up to 5) giving every fold >= 25 detections
folds <- choose_folds_min_detections(
  data = train_data,
  block_column = "spatio_temporal_block",
  class_column = "presence",
  min_detections = 25,
  max_folds = 5
)
print(folds)

# Add a fold column, assigned within each block x class group
train_data <- assign_spatial_cv_folds(
  data = train_data,
  block_column = "spatio_temporal_block",
  class_column = "presence",
  n_folds = folds
)

# Check: detections, absences, and blocks per fold
train_data %>%
  group_by(fold) %>%
  summarise(
    detections = sum(presence == 1),
    absences = sum(presence == 0),
    n_blocks = n_distinct(spatio_temporal_block),
    .groups = "drop"
  ) %>%
  print()

# Check: how many folds each block appears in (blocks with >= k rows)
train_data %>%
  group_by(spatio_temporal_block) %>%
  summarise(n_folds_in_block = n_distinct(fold), n = n(), .groups = "drop") %>%
  filter(n >= folds) %>%
  count(n_folds_in_block, name = "n_blocks") %>%
  print()

# Figure 3: share of each hex cell's training rows held out in fold 1,
# for the guide's folds and for leave-block-out folds
set.seed(7)

# Leave-block-out alternative: give each whole hex cell a single fold
# (the same shuffled 1, 2, ..., k sequence, applied to cells instead of rows)
cell_folds <- tibble(spatial_block = unique(train_data$spatial_block)) %>%
  mutate(lbo_fold = sample(rep_len(seq_len(folds), n())))

# Share of each cell's training rows that fall in fold 1 under each design
held_out <- train_data %>%
  left_join(cell_folds, by = "spatial_block") %>%
  group_by(spatial_block) %>%
  summarise(
    `Block-stratified (this guide)` = mean(fold == 1),
    `Leave-block-out` = mean(lbo_fold == 1),
    .groups = "drop"
  ) %>%
  tidyr::pivot_longer(-spatial_block, names_to = "design",
                      values_to = "share_held_out")

# Attach the shares to the hex polygons from Step 1
cv_map <- block_cells %>%
  inner_join(held_out, by = c("seqnum" = "spatial_block"))

p3 <- ggplot(cv_map) +
  geom_sf(aes(fill = share_held_out), colour = "white", linewidth = 0.3) +
  geom_sf(data = alberta_lcc, fill = NA, colour = "black", linewidth = 0.5) +
  facet_wrap(~design) +
  scale_fill_gradientn(
    colours = met.brewer("Hokusai3", type = "continuous", direction = -1),
    limits = c(0, 1),
    labels = function(x) paste0(x * 100, "%")
  ) +
  scale_x_continuous(breaks = c(-120, -115, -110)) +
  labs(x = NULL, y = NULL, fill = "Training rows held out in fold 1") +
  theme_science_map() +
  theme(
    strip.text = element_text(face = "bold"),
    legend.position = "bottom",
    legend.key.width = unit(1.5, "cm")
  )
ggsave("figures/cv_fold_designs.png", p3, width = 9, height = 6, dpi = 150)

# Check: range of the held-out share across cells under each design
cv_map %>%
  st_drop_geometry() %>%
  group_by(design) %>%
  summarise(min = min(share_held_out), median = median(share_held_out),
            max = max(share_held_out))

# 4. Draw stratified bootstrap resamples ----
cat("
== Step 4: bootstrap index matrix ==
")
# 50 replicates, each resampling every block x class group to its own
# size with replacement
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
# Check: rows x replicates
print(dim(bootstrap_ids))
# Check: share of distinct rows in one replicate (~63% for a bootstrap)
cat("Unique rows in replicate 1:",
    length(unique(bootstrap_ids[, 1])), "of", nrow(train_data), "\n")
# Check: replicates differ from each other
cat("Replicates identical to replicate 1:",
    sum(apply(bootstrap_ids, 2, identical, bootstrap_ids[, 1])), "of 50\n")

# 5. Use the folds and bootstrap resamples in a model ----
cat("
== Step 5: bootstrap model results ==
")
# AUC from the Mann-Whitney U statistic: the probability that a random
# detection gets a higher prediction than a random absence
auc <- function(obs, pred) {
  r <- rank(pred)
  n1 <- sum(obs == 1)
  n0 <- sum(obs == 0)
  (sum(r[obs == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# Held-out test set from Step 2
test_data <- split_data %>% filter(split == "test")

# One model per bootstrap replicate
boot_results <- map_dfr(seq_len(ncol(bootstrap_ids)), function(b) {
  # Rows for replicate b (duplicates keep their Step 3 fold)
  boot_data <- train_data[bootstrap_ids[, b], ]

  # Block-stratified CV: hold out each fold in turn
  cv_auc <- map_dbl(seq_len(folds), function(k) {
    fit <- glm(presence ~ latitude + longitude, family = binomial,
               data = filter(boot_data, fold != k))
    held_out <- filter(boot_data, fold == k)
    auc(held_out$presence, predict(fit, held_out, type = "response"))
  })

  # Refit on the full replicate and score on the test set
  fit <- glm(presence ~ latitude + longitude, family = binomial,
             data = boot_data)
  tibble(
    replicate = b,
    cv_auc = mean(cv_auc),
    test_auc = auc(test_data$presence,
                   predict(fit, test_data, type = "response")),
    latitude_coef = coef(fit)[["latitude"]]
  )
})

# Mean and spread of each metric across replicates
boot_results %>%
  tidyr::pivot_longer(-replicate, names_to = "metric") %>%
  group_by(metric) %>%
  summarise(mean = mean(value), sd = sd(value))
