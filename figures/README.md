# Supplementary Discussion 6.7.4 figures

This directory contains the figure assets used by `main.tex`.

## Completed assets

- `gcnm_synthetic_audit_example.png` is the full synthetic anatomy, conductivity, voltage, and waveform audit page used in Figure 1.
- `source_panels/` contains the four train, validation, and test audit pages assembled by LaTeX into the synthetic diversity gallery in Figure 2.
- `gcnm_parameter_selection.pdf` is the architecture and loss ablation used in the parameter-selection figure.

## Assets still to be generated

- `gcnm_synthetic_truth_newton_s1_s2.pdf` will replace the current synthetic comparison placeholder. It should contain synthetic ground truth, Newton, GCNM stage 1, and GCNM stage 2.
- `gcnm_subject_beat_atlas.tex` will replace the current participant-atlas placeholder. It should contain one page per participant with ten Newton images above the ten matched GCNM stage-2 images.
- `gcnm_synthetic_diversity.pdf` is an optional polished replacement for the four-panel gallery assembled directly in `main.tex`.

The manuscript compiles without the unfinished files because `main.tex` uses `\IfFileExists` and displays explicit placeholders until those assets are added.
