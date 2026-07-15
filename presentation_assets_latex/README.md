# PVI-GCNM standalone LaTeX results assets

This directory is a presentation-ready asset pack, separate from the technical
report. Every `.tex` file is a self-contained `standalone` document: compile it
to obtain a tightly cropped PDF that can be inserted into Beamer, PowerPoint,
Keynote, or another LaTeX document. The report itself is not modified.

## Render everything

From this directory:

```bash
./build_all.sh
```

The compiled PDFs are written to `build/tables`, `build/charts`, and
`build/galleries`. To compile one asset manually, run Tectonic from that
asset's source directory so its relative imports resolve, for example:

```bash
cd charts
XDG_CACHE_HOME=/tmp/tectonic-cache \
XDG_CONFIG_HOME=/tmp/tectonic-config \
  tectonic --outdir ../build/charts chart_02_synthetic_metrics.tex
```

## Asset inventory

| Output | Source evidence | Intended use |
|---|---|---|
| `table_01_problem_scale.pdf` | Report Tables 1--2 and Figure 1 | Measurement/unknown scale |
| `table_02_method_comparison.pdf` | Report Table 3 plus the faithful implementation | Three-way method contrast |
| `table_03_synthetic_accuracy.pdf` | Report Tables 8 and 10 | Headline exact-nonlinear holdout result |
| `table_04_real_subject6.pdf` | Report Table 13 | Honest real-data transfer result |
| `table_05_ablation_summary.pdf` | Report Table 10 | One-factor ablation appendix |
| `table_06_production_parity.pdf` | Report Table 4 | Python/MATLAB plumbing validation |
| `chart_01_measurement_deficit.pdf` | Report Figure 1 | 928 versus 32 measurements |
| `chart_02_synthetic_metrics.pdf` | Report Tables 8 and 10 | Baseline/physics/learned ladder |
| `chart_03_ablation_sweep.pdf` | Report Figure 7 and Table 10 | Selection path for alpha and beta |
| `chart_04_voltage_residual.pdf` | Report Table 13 and evaluation summaries | Synthetic/real residual by stage |
| `chart_05_training_curves.pdf` | Report Figure 3 | Optional training appendix asset |
| `gallery_a_method_evolution.pdf` | Report Figure 8 arrays, exact four-column regeneration | Earlier fixed-feature limitation |
| `gallery_b_selected_best_median_worst.pdf` | Report Figure 10 | Selected model best/median/worst |
| `gallery_c_real_pseudo_reference.pdf` | Report Figure 12 | Held-out real subject-6 stages |
| `gallery_d_cross_domain.pdf` | Report Figure 11 | Descriptive smoke-pack comparison |

## Scientific reading rules encoded in the assets

- Synthetic assets use known clean `Delta sigma*` ground truth and report true
  reconstruction metrics.
- Real subject-6 assets use a PVI Newton reconstruction only as a
  **pseudo-reference**. They never label it as anatomical truth or call the
  resulting correlations accuracy.
- Gray means Newton/PVI baseline, teal means physics-only iterative LM, blue
  means the learned GCNM, amber marks oracle/disqualified conditions, and
  coral marks real-data pseudo-reference evidence.
- Metric headers include direction arrows; winning table cells are bold.
- Galleries retain the registration, error maps, and objective sample selection
  embedded in the report-generated source figures. Gallery A contains its
  original colorbar; the wrappers for B--D add explicit flat-binned conductivity
  keys and the exact symmetric limits used by the plotting code.

## Source images and reproducibility boundary

The gallery wrappers import frozen PDFs from `source_images/`: B--D are copies
of the report figures, while A is regenerated from the same saved arrays with
the requested four-column layout. They do not rerun training or change any model configuration.
The numerical tables and PGFPlots charts are declarative LaTeX using the frozen
evaluation values for checkpoint `faithful_background025_hom_seed0`, the 32
exact-nonlinear holdout phantoms, and the 1,600 held-out subject-6 phases from
trials 10--11.

Gallery A intentionally removes the extra historical smoke-model column present
in the archived report figure so that it follows the four-column specification.
Regenerate it from the saved nonlinear-evaluation arrays with
`scripts/generate_gallery_a.py`; this is plotting only and does not run a model.

One correction to the supplied build list is intentional: the production PVI
row in Table 2 uses the code-backed homogeneous baseline
`sigma_0 = 0.7 S/m`, not `0.71 S/m`.

No authentic SRL logo asset was present in the repositories searched. The
shared style therefore renders a small `SRL` text placeholder. To use the
official mark, redefine `\SRLLogo` before `\input{../asset_style.tex}` in each
standalone source, or replace the placeholder definition in `asset_style.tex`.

## Files to edit

- `asset_style.tex`: colors, typography, title treatment, tags, table helpers,
  and common PGFPlots defaults.
- `tables/*.tex`: exact table rows, task lines, metric glossaries, and captions.
- `charts/*.tex`: chart data and axis settings.
- `galleries/*.tex`: gallery framing, warnings, and captions.
- `source_images/*.pdf`: visual panels derived from the frozen evaluation
  artifacts (B--D are report copies; A is the exact four-column regeneration).

On the Lakeshore environment, `asset_style.tex` uses the installed DejaVu Sans
and Nimbus Mono PS files to match the deck. On another machine it falls back to
the LaTeX installation's default sans and mono families if those paths are absent.
