# Spatio-temporal stratification of survey locations: worked example
# Run from this vignette's folder so the relative source() paths resolve.

library(dplyr)
library(purrr)
library(ggplot2)
library(sf)
library(dggridR)
library(rnaturalearth)
library(MetBrewer)
library(utilR) # theme_science_map()

source("R/stratified_split.R")
source("R/assign_spatial_cv_folds.R")
source("R/assign_bootstrap_ids_matrix.R")

# 0. Simulate survey data ----
# 10,000 survey locations within Alberta, 2000-2024.
# Detection probability rises toward the south-east so that
# some blocks have many detections and others have none.
alberta <- ne_states(country = "Canada", returnclass = "sf") %>%
  filter(name_en == "Alberta")

set.seed(42)
n <- 10000
locations <- st_sample(alberta, size = n) %>%
  st_coordinates()

data <- tibble(
  longitude = locations[, "X"],
  latitude  = locations[, "Y"],
  year      = sample(2000:2024, n, replace = TRUE)
) %>%
  mutate(
    p_det    = plogis(-3.5 + 0.35 * (60 - latitude) / 2 +
                        0.2 * (longitude + 120) / 2),
    presence = rbinom(n, 1, p_det)
  ) %>%
  select(-p_det)

# 1. Build spatial and temporal blocks ----
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
    spatio_temporal_block = interaction(
      droplevels(as.factor(spatial_block)),
      droplevels(as.factor(temporal_block)),
      sep = "::",
      drop = TRUE
    )
  ) %>%
  group_by(spatio_temporal_block) %>%
  mutate(
    spatio_temporal_block = paste0(
      "block_", cur_group_id()
    )
  ) %>%
  ungroup()

cat("\n== Step 1: block summary ==\n")
data %>%
  summarise(
    n_obs = n(),
    n_detections = sum(presence),
    n_spatial_blocks = n_distinct(spatial_block),
    n_spatio_temporal_blocks = n_distinct(spatio_temporal_block)
  ) %>%
  print()
print(table(data$temporal_block))

# Map the blocks
dir.create("figures", showWarnings = FALSE)

alberta_lcc <- st_transform(alberta, crs = 3978)

block_cells <- dgcellstogrid(grid, cells = unique(data$spatial_block)) %>%
  st_as_sf() %>%
  mutate(seqnum = as.integer(seqnum)) %>%
  st_transform(crs = 3978)

spatial_counts <- block_cells %>%
  left_join(count(data, spatial_block, name = "n_obs"),
            by = c("seqnum" = "spatial_block"))

p1 <- ggplot(spatial_counts) +
  geom_sf(aes(fill = n_obs), colour = "white", linewidth = 0.3) +
  geom_sf(data = alberta_lcc, fill = NA, colour = "black", linewidth = 0.5) +
  scale_fill_met_c("Hokusai3", direction = -1) +
  labs(x = NULL, y = NULL, fill = "Locations per block") +
  theme_science_map()
ggsave("figures/spatial_blocks.png", p1, width = 6, height = 6, dpi = 150)

st_counts <- block_cells %>%
  inner_join(count(data, spatial_block, temporal_block, name = "n_obs"),
             by = c("seqnum" = "spatial_block"))

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
split_data <- stratified_split(
  data = data,
  block_column = "spatio_temporal_block",
  class_column = "presence",
  train_frac = 0.75
)
cat("\n== Step 2: split by class ==\n")
count(split_data, spatio_temporal_block, presence, split)
table(split_data$split, split_data$presence)

train_data <- split_data %>% filter(split == "train") %>% select(-split)

# 3. Choose the number of folds and assign them ----
folds <- choose_folds_min_detections(
  data = train_data,
  block_column = "spatio_temporal_block",
  class_column = "presence",
  min_detections = 25,
  max_folds = 5
)
cat("\n== Step 3: folds chosen ==\n")
print(folds)

train_data <- assign_spatial_cv_folds(
  data = train_data,
  block_column = "spatio_temporal_block",
  class_column = "presence",
  n_folds = folds
)

cat("\n== Step 3: detections and absences per fold ==\n")
train_data %>%
  group_by(fold) %>%
  summarise(
    detections = sum(presence == 1),
    absences = sum(presence == 0),
    n_blocks = n_distinct(spatio_temporal_block),
    .groups = "drop"
  ) %>%
  print()

cat("\n== Step 3: folds represented per block ==\n")
train_data %>%
  group_by(spatio_temporal_block) %>%
  summarise(n_folds_in_block = n_distinct(fold), n = n(), .groups = "drop") %>%
  filter(n >= folds) %>%
  count(n_folds_in_block, name = "n_blocks") %>%
  print()

# 4. Draw stratified bootstrap resamples ----
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
cat("\n== Step 4: bootstrap index matrix ==\n")
print(dim(bootstrap_ids))
cat("Unique rows in replicate 1:",
    length(unique(bootstrap_ids[, 1])), "of", nrow(train_data), "\n")
cat("Replicates identical to replicate 1:",
    sum(apply(bootstrap_ids, 2, identical, bootstrap_ids[, 1])), "of 50\n")

# 5. Use the folds and bootstrap resamples in a model ----
auc <- function(obs, pred) {
  r <- rank(pred)
  n1 <- sum(obs == 1)
  n0 <- sum(obs == 0)
  (sum(r[obs == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

test_data <- split_data %>% filter(split == "test")

boot_results <- map_dfr(seq_len(ncol(bootstrap_ids)), function(b) {
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

cat("\n== Step 5: bootstrap model results ==\n")
boot_results %>%
  tidyr::pivot_longer(-replicate, names_to = "metric") %>%
  group_by(metric) %>%
  summarise(mean = mean(value), sd = sd(value))
