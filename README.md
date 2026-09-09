# 2D PVI-GCNM

This repository owns the graph-convolutional Newton reconstruction method for
two-dimensional peripheral vascular impedance imaging. Its scope is the GCNM
itself: finite-element physics, mesh and graph utilities, synthetic supervision,
GCNM training/evaluation, checkpoints, and the deployable reconstruction API.

The real-PVI-to-blood-pressure workflow is maintained separately in
`pvi_gcnm_bp_pipeline`. That repository may import the public `gcnm_pvi`
package; this repository must never import the BP pipeline package.

## Method boundary

At reconstruction stage `k`, the implementation recomputes the nonlinear
forward solution, voltage residual, Jacobian, and regularized physics update at
the current conductivity estimate. A stage-specific graph network maps the
current estimate, physics update, and configured spatial features to the next
estimate. The two learned stages have independent parameters.

The public inference boundary is intentionally separate from the training
entrypoints:

- `gcnm_pvi.coordinate_runtime` builds and evaluates coordinate GCN graphs;
- `gcnm_pvi.vessel_runtime` builds and evaluates vessel-model graphs;
- `gcnm_pvi.representations` exposes the reconstruction representations used by
  downstream consumers;
- `gcnm_pvi.electrode_protocol` defines the canonical 8-electrode,
  32-measurement protocol without depending on real-data ingestion code.

## Repository contents

```text
gcnm_pvi/                 physics, models, simulation, training, evaluation, runtime
configs/                  ring and GCNM experiment configurations
scripts/                  GCNM generation, training, evaluation, and report entrypoints
slurm/                    cluster launchers for GCNM work
tests/                    numerical, runtime-boundary, and regression tests
data/ring_meshes/         small versioned mesh contracts
reports/gcnm_pvi_latex/   mathematical report source and publication figures
```

Participant data, generated synthetic corpora, checkpoints, Parquet caches,
artifacts, exports, scheduler logs, and Python environments are not committed.

## Installation

Python 3.10 or newer and the external PVI solver are required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'

export PVI_SOLVER_ROOT=/path/to/Peripheral-Vascular-Impedance-Imaging/python_port/pvi_solver
export GCNM_PYTHON="$PWD/.venv/bin/python"
export PYTHONPATH="$PWD:$PVI_SOLVER_ROOT:${PYTHONPATH:-}"
```

On the cluster, copy `cluster.env.example` to the ignored
`env/cluster.env` and set the real paths there.

## Verification

```bash
"$GCNM_PYTHON" -m compileall -q gcnm_pvi
"$GCNM_PYTHON" -m pytest -q
bash -n scripts/*.sh slurm/*.sh
```

The regression suite covers nonlinear stage recomputation, physics signs,
graph-output contracts, dual-mesh behavior, public runtime parity, electrode
configuration, synthetic data generation, and checkpoint/evaluation behavior.

## GCNM workflows

The retained launchers cover the historical subject-6 experiments and the
ring-specific synthetic training used by the production reconstructions. The
main scientific sequence is:

1. export or validate the forward/inverse meshes and image mapping;
2. generate clean synthetic conductivity targets and simulated voltages;
3. train the two GCNM stages sequentially;
4. recompute the nonlinear physics after stage 1 before fitting stage 2;
5. evaluate on exact nonlinear holdouts and export versioned checkpoints and
   reconstruction figures.

The selected configuration uses a homogeneous `0.7 S/m` inverse baseline,
three hidden graph-convolutional layers with 64 channels, direct stage output,
coordinate features, and the documented target/background loss weighting. The
experiment configurations and saved manifests remain the authority for an
individual run.

## Storage during the repository split

Existing generated data remain at their established locations under this
checkout. They have not been copied, renamed, or deleted. The BP repository
references those persistent roots explicitly, while its precomputed Parquet
datasets remain under `/mmfs1/scratch/$USER/gcnm_population_6ch`.

The recoverable pre-split source states are:

- `safety/pre-split-head-20260908` — original tracked repository state;
- `repo-split/preparation-20260908` — reviewed separation work.

No generated-data directory is part of the source-removal step.

## Documentation

- `REPOSITORY_BOUNDARY.md` — authoritative ownership, dependency, storage, and recovery contract
- `ANATOMICAL_GCNM.md` — anatomical phantom and clean-supervision design
- `SUBJECT006_STATUS.md` — subject-6 geometry and reconstruction evidence
- `IMPLEMENTATION_REPORT.md` — implementation and verification history
- `PVI_GCNM_ADAPTATION_PLAN.md` — original method-adaptation analysis
- `reports/gcnm_pvi_latex/main.pdf` — derivation, experiments, and limitations

The BP/Parquet/CRT/SAMBA operating instructions now live in the separate BP
pipeline repository.
