# Method guides

Method guides (vignettes) that reconstruct analysis workflows from project code. Each guide walks through the steps, key code, decision points, and limitations of one method, and runs a reproducible example on simulated data.

## Vignettes

| Vignette | Method | Source project |
|---|---|---|
| [Spatio-temporal stratification of survey locations](spatio_temporal_blocking/README.md) | Hexagonal spatio-temporal blocks and block-stratified training–test splits, CV folds, and bootstrap resamples for presence–absence models | Adapted from `invasive_species_indicator` |

Each vignette folder contains the guide, an `example.R` script that reproduces its outputs on simulated data, the helper functions in `R/`, and the figures. Source code is on [GitHub](https://github.com/bgcasey/method_guides).
