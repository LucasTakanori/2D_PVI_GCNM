# GCNM--PVI technical report

The editable report is `main.tex`; the compiled deliverable is `main.pdf`.

The report figures are generated directly from the saved subject-6 experiment
artifacts by `make_figures.py`. To regenerate them:

```bash
MPLCONFIGDIR=/tmp/gcnm-mpl \
PVI_SUBJECT006_H5=/path/to/subject006_baseline_masked.h5 \
  "$GCNM_PYTHON" make_figures.py
```

Figure regeneration requires the untracked subject-6 ring mapping, anatomical
prediction NPZ files, training/evaluation JSON reports, previous smoke output,
and the private HDF5 named by `PVI_SUBJECT006_H5`. These scientific artifacts are
intentionally not distributed in the public code repository. The committed PDF
figures and `main.pdf` can be read and recompiled without them.

Faithful experiment figures are generated separately from the completed ablation
and real-PVI reports:

```bash
FINAL_FAITHFUL_EXPERIMENT=faithful_background025_hom_seed0 \
  MPLCONFIGDIR=/tmp/gcnm-mpl "$GCNM_PYTHON" make_faithful_figures.py
```

To compile the document with the locally installed Tectonic executable:

```bash
XDG_CACHE_HOME=/tmp/tectonic-cache \
XDG_CONFIG_HOME=/tmp/tectonic-config \
  tectonic \
  --keep-logs --keep-intermediates main.tex
```

Run both commands from this directory. The first Tectonic compilation may need
network access to retrieve its standard TeX bundle. The current compiled report
has 30 letter-size pages. It includes the original-method audit, earlier
fixed-feature results, implemented faithful equations, complete core/baseline/loss
tables, best/median/worst clean-ground-truth output comparisons, and the full-test
real-PVI pseudo-reference gallery. The faithful figures written by
`make_faithful_figures.py` are:

- `faithful_core_comparison.pdf`
- `faithful_baseline_comparison.pdf`
- `faithful_ablation_comparison.pdf`
- `faithful_output_gallery.pdf`
- `faithful_real_pvi_gallery.pdf`
