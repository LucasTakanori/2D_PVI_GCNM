# Repository boundary

This repository owns the GCNM reconstruction method: finite-element physics,
mesh and graph utilities, synthetic supervision, GCNM training and evaluation,
ring-specific checkpoints, and the public reconstruction runtime.

The sibling `pvi_gcnm_bp_pipeline` repository owns real PVI ingestion,
participant/sample splits, GCNM representation export, Parquet materialization,
CRT/SAMBA training, BP inference, and `pvi_ml`-compatible artifacts.

Dependency direction is one way:

```text
pvi_gcnm_bp  ->  gcnm_pvi  ->  external PVI solver
```

The core package must never import `pvi_gcnm_bp`. Shared electrode and runtime
contracts therefore live here in `gcnm_pvi.electrode_protocol`,
`gcnm_pvi.coordinate_runtime`, `gcnm_pvi.vessel_runtime`, and
`gcnm_pvi.representations`.

## Storage boundary

Repository separation changes source ownership, not generated-data locations.
Existing `data`, `models`, `artifacts`, `exports`, `figures`, `reports`, and
`logs` roots remain under this checkout. Precomputed BP Parquet datasets remain
under `/mmfs1/scratch/$USER/gcnm_population_6ch` and are referenced by the BP
repository through explicit environment variables.

## Recovery points

- `safety/pre-split-head-20260908` preserves the original tracked repository.
- `repo-split/preparation-20260908` contains the reviewed separation history.
- The BP repository has its own `safety/extraction-source-tree` branch.

No generated data are removed by the source split.
