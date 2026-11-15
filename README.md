# method_guides

Method guides (vignettes) that reconstruct analysis workflows from project code. Each guide walks through the steps, key code, decision points, and limitations of one method, and runs a reproducible example on simulated data.

## Vignettes

| Vignette | Method | Source project |
|---|---|---|
| [Spatio-temporal stratification of survey locations](vignettes/spatio_temporal_blocking/spatio_temporal_blocking.md) | Hexagonal spatio-temporal blocks and block-stratified training–test splits, CV folds, and bootstrap resamples for presence–absence models | Adapted from `invasive_species_indicator` |

## Layout

Each vignette lives in its own folder under `vignettes/`:

```
vignettes/<name>/
├── <name>.md     # the guide
├── example.R     # simulated-data example that reproduces the outputs in the guide
├── R/            # helper functions copied from the source project
└── figures/      # figures referenced by the guide
```

Run `example.R` from inside its vignette folder so the relative `source("R/...")` paths resolve.
