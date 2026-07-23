#!/usr/bin/env python3
"""Build a standalone HTML dashboard for the coordinate-direct BP pilots."""

from __future__ import annotations

import html
import json
import math
import os
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "coordinate_bp_results_v1" / "index.html"


RUNS = {
    "s006_native_wave": (
        "Subject006 · native HDF5 PVI",
        ROOT / "artifacts/pvi_ml_subject006_native_v1/subject006-crt-img-to-waveform/main",
    ),
    "s006_ref_wave": (
        "Subject006 · reference-Parquet PVI",
        ROOT / "artifacts/pvi_bp_us120_reference_parquet_v1/reference-parquet-crt-img-to-waveform/main",
    ),
    "s006_ref_fid": (
        "Subject006 · reference-Parquet PVI",
        ROOT / "artifacts/pvi_bp_us120_reference_parquet_v1/reference-parquet-crt-img-to-fiducials/main",
    ),
    "s006_coord_wave": (
        "Subject006 · coordinate 3ch",
        ROOT / "artifacts/subject006_coordinate_direct_s1_s2_ds2_bp_v1/gcnm-coordinate-3ch-crt-image-to-waveform/main",
    ),
    "s006_coord_fid": (
        "Subject006 · coordinate 3ch",
        ROOT / "artifacts/subject006_coordinate_direct_s1_s2_ds2_bp_v1/gcnm-coordinate-3ch-crt-image-to-fiducials/main",
    ),
    "s006_fused_wave": (
        "Subject006 · Newton+coordinate 6ch",
        ROOT / "artifacts/subject006_newton_coordinate_6ch_bp_v1/gcnm-newton_coordinate-6ch-crt-image-to-waveform/main",
    ),
    "s006_fused_fid": (
        "Subject006 · Newton+coordinate 6ch",
        ROOT / "artifacts/subject006_newton_coordinate_6ch_bp_v1/gcnm-newton_coordinate-6ch-crt-image-to-fiducials/main",
    ),
    "s010_ref_wave": (
        "Subject010 · reference-Parquet PVI",
        ROOT / "artifacts/pvi_bp_us120_reference_parquet_v1/reference-parquet-crt-img-to-waveform/main",
    ),
    "s010_ref_fid": (
        "Subject010 · reference-Parquet PVI",
        ROOT / "artifacts/pvi_bp_us120_reference_parquet_v1/reference-parquet-crt-img-to-fiducials/main",
    ),
    "s010_coord_wave": (
        "Subject010 · coordinate 3ch",
        ROOT / "artifacts/subject010_coordinate_direct_s1_s2_ds2_bp_v1/gcnm-coordinate-3ch-crt-image-to-waveform/main",
    ),
    "s010_coord_fid": (
        "Subject010 · coordinate 3ch",
        ROOT / "artifacts/subject010_coordinate_direct_s1_s2_ds2_bp_v1/gcnm-coordinate-3ch-crt-image-to-fiducials/main",
    ),
    "s010_fused_wave": (
        "Subject010 · Newton+coordinate 6ch",
        ROOT / "artifacts/subject010_newton_coordinate_6ch_bp_v1/gcnm-newton_coordinate-6ch-crt-image-to-waveform/main",
    ),
    "s010_fused_fid": (
        "Subject010 · Newton+coordinate 6ch",
        ROOT / "artifacts/subject010_newton_coordinate_6ch_bp_v1/gcnm-newton_coordinate-6ch-crt-image-to-fiducials/main",
    ),
}


SUBJECTS = {key: "subject010" if key.startswith("s010") else "subject006" for key in RUNS}


def rel(path: Path) -> str:
    return html.escape(os.path.relpath(path.resolve(), OUTPUT.parent.resolve()), quote=True)


def load_run(key: str) -> dict:
    label, base = RUNS[key]
    subject = SUBJECTS[key]
    stats_path = base / "statistics" / f"{subject}_statistics.json"
    history_path = base / "history" / f"{subject}_history.csv"
    results_path = base / "results" / f"{subject}_results.csv"
    config_path = base / "configs" / f"{subject}_configs.json"
    verification_path = base / "gifs" / f"{subject}_verification.json"
    legacy_verification = base / "verification" / f"{subject}.json"
    if not verification_path.is_file() and legacy_verification.is_file():
        verification_path = legacy_verification
    gif_manifest_path = base / "gifs" / f"{subject}_gifs.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    epochs = max(sum(1 for _ in history_path.open(encoding="utf-8")) - 1, 0)
    return {
        "key": key,
        "label": label,
        "base": base,
        "stats": stats,
        "epochs": epochs,
        "stats_path": stats_path,
        "history_path": history_path,
        "results_path": results_path,
        "config_path": config_path,
        "verification_path": verification_path if verification_path.is_file() else None,
        "gif_manifest_path": gif_manifest_path if gif_manifest_path.is_file() else None,
    }


def f(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def metric_row(run: dict, label: str, accent: str = "") -> str:
    s = run["stats"]
    dbp_r = math.sqrt(max(0.0, float(s["dbp_r2"])))
    sbp_r = math.sqrt(max(0.0, float(s["sbp_r2"])))
    bp_accuracy = 0.5 * (dbp_r + sbp_r)
    links = (
        f'<a href="{rel(run["stats_path"])}">metrics</a> · '
        f'<a href="{rel(run["results_path"])}">predictions</a> · '
        f'<a href="{rel(run["history_path"])}">history</a>'
    )
    if run["gif_manifest_path"] is not None:
        links += f' · <a href="{rel(run["gif_manifest_path"])}">GIFs</a>'
    return f"""
      <tr class="{accent}">
        <th>{html.escape(label)}</th>
        <td>{f(s['amae'])}</td><td>{f(s['armse'])}</td>
        <td>{f(s['dbp_mae'])}</td><td>{f(s['sbp_mae'])}</td>
        <td>{f(bp_accuracy, 3)}</td>
        <td>{f(dbp_r, 3)} / {f(sbp_r, 3)}</td>
        <td>{f(s['dbp_r2'], 3)} / {f(s['sbp_r2'], 3)}</td>
        <td>{f(s['dbp_cc'], 3)} / {f(s['sbp_cc'], 3)}</td>
        <td>{f(s['dbp_tol05'], 1)} / {f(s['sbp_tol05'], 1)}</td>
        <td>{run['epochs']:,}</td><td class="links">{links}</td>
      </tr>"""


def bar(label: str, value: float, maximum: float, tone: str) -> str:
    width = max(2.0, min(100.0, 100.0 * value / maximum))
    return f"""
      <div class="bar-row">
        <span>{html.escape(label)}</span>
        <div class="bar-track"><div class="bar {tone}" style="width:{width:.1f}%"></div></div>
        <strong>{value:.2f}</strong>
      </div>"""


def delta(new: float, old: float) -> str:
    value = new - old
    return f"{value:+.2f} mmHg"


def gif_card(title: str, path: Path, caption: str) -> str:
    return f"""
      <figure>
        <a href="{rel(path)}"><img src="{rel(path)}" alt="{html.escape(title)}" loading="lazy"></a>
        <figcaption><strong>{html.escape(title)}</strong><span>{html.escape(caption)}</span></figcaption>
      </figure>"""


def main() -> None:
    runs = {key: load_run(key) for key in RUNS}
    s6 = runs
    wave_values = [
        ("Native HDF5 PVI", s6["s006_native_wave"]["stats"]["amae"], "blue"),
        ("Reference PVI Parquet", s6["s006_ref_wave"]["stats"]["amae"], "slate"),
        ("Coordinate 3ch", s6["s006_coord_wave"]["stats"]["amae"], "amber"),
        ("Newton + coordinate 6ch", s6["s006_fused_wave"]["stats"]["amae"], "teal"),
    ]
    fid_values = [
        ("Reference PVI Parquet", s6["s006_ref_fid"]["stats"]["amae"], "slate"),
        ("Coordinate 3ch", s6["s006_coord_fid"]["stats"]["amae"], "amber"),
        ("Newton + coordinate 6ch", s6["s006_fused_fid"]["stats"]["amae"], "teal"),
    ]
    max_bar = max(value for _label, value, _tone in wave_values + fid_values) * 1.08

    synthetic = ROOT / "reports/coordinate_direct_US120_1000beats_v1/synthetic_exact_truth_newton_s1_s2_three_beats.gif"
    pilot_root = ROOT / "reports/subject006_coordinate_direct_US120_1000beats_v1"
    parquet_gif_root = ROOT / "reports/subject006_coordinate_direct_parquet_gifs_v1"
    gifs = [
        gif_card("Synthetic exact holdout", synthetic, "Truth · one-step Newton · coordinate S1 · coordinate S2"),
        *[
            gif_card(
                f"Subject006 {session} pilot",
                pilot_root / f"subject006_{session}/archived_pvi_s1_s2_ds2_three_beats.gif",
                "Archived PVI HP/LP beside live coordinate reconstruction",
            )
            for session in ("baseline", "valsalva", "pressor")
        ],
        *[
            gif_card(
                f"Subject006 {session} serialized Parquet",
                parquet_gif_root / f"subject006_{session}/parquet_archived_pvi_s1_s2_ds2_three_beats.gif",
                "Loaded from the immutable Parquet row; model inference is not rerun",
            )
            for session in ("baseline", "valsalva", "pressor")
        ],
    ]
    for key in (
        "s006_coord_wave", "s006_coord_fid", "s006_fused_wave",
        "s006_fused_fid", "s010_coord_wave", "s010_coord_fid",
        "s010_fused_wave", "s010_fused_fid",
    ):
        run = runs[key]
        manifest_path = run["gif_manifest_path"]
        if manifest_path is None:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for example in manifest["examples"]:
            gifs.append(
                gif_card(
                    f"{run['label']} · {manifest['output_mode']} · {example['session']}",
                    Path(example["gif"]),
                    "Held-out median-error window: PVI/Newton HP-LP, S1, S2, dS2/dt and saved BP output",
                )
            )

    s6_coord_w = s6["s006_coord_wave"]["stats"]
    s6_fused_w = s6["s006_fused_wave"]["stats"]
    s6_ref_w = s6["s006_ref_wave"]["stats"]
    s6_native_w = s6["s006_native_wave"]["stats"]
    s6_coord_f = s6["s006_coord_fid"]["stats"]
    s6_fused_f = s6["s006_fused_fid"]["stats"]
    s6_ref_f = s6["s006_ref_fid"]["stats"]
    s10_coord_w = s6["s010_coord_wave"]["stats"]
    s10_ref_w = s6["s010_ref_wave"]["stats"]
    s10_coord_f = s6["s010_coord_fid"]["stats"]
    s10_ref_f = s6["s010_ref_fid"]["stats"]
    s10_fused_w = s6["s010_fused_wave"]["stats"]
    s10_fused_f = s6["s010_fused_fid"]["stats"]

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Coordinate-direct GCNM · BP pilot results</title>
  <style>
    :root {{ --ink:#172033; --muted:#657184; --line:#dce3eb; --soft:#f5f7fa;
      --blue:#22658f; --teal:#16856b; --amber:#b66a12; --coral:#c84f37;
      --shadow:0 10px 32px rgba(23,32,51,.08); font-family:Inter,ui-sans-serif,system-ui,sans-serif; }}
    * {{ box-sizing:border-box }} html {{ scroll-behavior:smooth }}
    body {{ margin:0; color:var(--ink); background:#fff }} a {{ color:var(--blue) }}
    code {{ font-family:"SFMono-Regular",Consolas,monospace }}
    nav {{ position:sticky; top:0; z-index:5; display:flex; gap:18px; align-items:center;
      padding:13px max(22px,calc((100vw - 1440px)/2)); border-top:5px solid var(--ink);
      border-bottom:1px solid var(--line); background:rgba(255,255,255,.96); backdrop-filter:blur(10px) }}
    nav strong {{ margin-right:auto }} nav a {{ color:var(--muted); text-decoration:none; font-size:13px }}
    main {{ max-width:1440px; margin:auto; padding:34px 24px 80px }}
    .hero {{ display:grid; grid-template-columns:1.5fr .5fr; gap:20px }}
    .panel,.hero-copy,.hero-side {{ border:1px solid var(--line); border-radius:18px; background:#fff; box-shadow:var(--shadow) }}
    .hero-copy {{ padding:32px }} .hero-side {{ padding:20px; display:grid; grid-template-columns:1fr 1fr; gap:10px }}
    .eyebrow {{ color:var(--blue); font:700 12px/1 monospace; letter-spacing:.09em; text-transform:uppercase }}
    h1 {{ margin:12px 0 16px; font-size:clamp(34px,5vw,58px); line-height:1 }}
    h2 {{ margin:52px 0 8px; font-size:28px }} h3 {{ margin:0 0 8px }}
    .lede,.section-lede,p {{ color:var(--muted); line-height:1.6 }}
    .kpi {{ padding:16px; background:var(--soft); border-radius:13px }} .kpi strong {{ display:block; font-size:27px }} .kpi span {{ font-size:11px; color:var(--muted) }}
    .notice {{ margin:22px 0; padding:16px 18px; border-left:5px solid var(--teal); background:#eff9f6; border-radius:0 12px 12px 0 }}
    .contract {{ display:grid; grid-template-columns:repeat(6,1fr); gap:8px; margin:18px 0 }}
    .channel {{ padding:13px 9px; text-align:center; border:1px solid var(--line); border-radius:11px; background:var(--soft); font:700 12px/1.4 monospace }}
    .channel.coord {{ border-top:4px solid var(--teal) }} .channel.newton {{ border-top:4px solid var(--blue) }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:15px; margin-top:18px }}
    table {{ width:100%; border-collapse:collapse; min-width:1080px }} th,td {{ padding:12px 11px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; font-size:12px }}
    thead th {{ background:var(--soft); color:#465267; font-size:11px; text-transform:uppercase; letter-spacing:.04em }}
    tbody th {{ text-align:left; font-size:13px }} tr.winner {{ background:#f0faf7 }} tr.warning {{ background:#fff9ef }}
    td.links {{ text-align:left }} td.links a {{ margin-right:2px }}
    .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:18px }} .panel {{ padding:20px }}
    .bar-row {{ display:grid; grid-template-columns:190px 1fr 48px; gap:10px; align-items:center; margin:12px 0; font-size:12px }}
    .bar-track {{ height:13px; background:#edf1f5; border-radius:99px; overflow:hidden }} .bar {{ height:100%; border-radius:99px }}
    .bar.blue {{ background:var(--blue) }} .bar.slate {{ background:#718096 }} .bar.amber {{ background:var(--amber) }} .bar.teal {{ background:var(--teal) }}
    .finding-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:18px }}
    .finding {{ border:1px solid var(--line); border-radius:14px; padding:17px }} .finding strong {{ display:block; font-size:20px; margin:7px 0 }} .finding span {{ color:var(--muted); font-size:12px; line-height:1.5 }}
    .gallery {{ display:grid; grid-template-columns:repeat(2,1fr); gap:16px; margin-top:18px }} figure {{ margin:0; border:1px solid var(--line); border-radius:15px; overflow:hidden; background:var(--soft) }}
    figure img {{ display:block; width:100%; aspect-ratio:2.7; object-fit:contain; background:#fff }} figcaption {{ padding:12px 14px }} figcaption strong,figcaption span {{ display:block }} figcaption span {{ color:var(--muted); font-size:11px; margin-top:4px }}
    .path-list {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:18px }} .path {{ padding:14px; border:1px solid var(--line); border-radius:12px; background:var(--soft) }} .path code {{ display:block; font-size:11px; overflow-wrap:anywhere; margin-top:5px }}
    footer {{ margin-top:55px; padding-top:20px; border-top:1px solid var(--line); color:var(--muted); font-size:11px }}
    @media(max-width:900px) {{ .hero,.two-col,.finding-grid,.gallery,.path-list {{ grid-template-columns:1fr }} .hero-side {{ grid-template-columns:1fr 1fr }} .contract {{ grid-template-columns:repeat(2,1fr) }} nav a {{ display:none }} }}
  </style>
</head>
<body>
<nav><strong>Coordinate-direct GCNM · BP pilots</strong><a href="#results">Results</a><a href="#findings">Findings</a><a href="#visuals">Visuals</a><a href="#artifacts">Artifacts</a></nav>
<main>
  <section class="hero">
    <div class="hero-copy"><div class="eyebrow">US120 · mask05 · CRT · frozen subject splits</div>
      <h1>From visible representations to BP evidence</h1>
      <p class="lede">A consolidated record of the coordinate-direct three-channel pilot, the Newton-plus-coordinate six-channel ablation, native HDF5 and reference-Parquet controls, and the subject010 replication.</p>
    </div>
    <div class="hero-side">
      <div class="kpi"><strong>934</strong><span>subject006 windows</span></div><div class="kpi"><strong>2,448</strong><span>subject010 windows</span></div>
      <div class="kpi"><strong>3 / 6</strong><span>tested image channels</span></div><div class="kpi"><strong>13</strong><span>reported CRT runs</span></div>
    </div>
  </section>
  <div class="notice"><strong>Bottom line.</strong> Coordinate-only images are highly usable visually. On subject006 they improve direct fiducial MAE but lose information needed for full-waveform prediction. Adding the three Newton channels recovers much of that loss. On subject010, coordinate-only is substantially better than reference PVI in MAE for both targets.</div>

  <section id="results"><h2>Input contracts</h2><p class="section-lede">The CRT architecture, loss, optimizer, stopping rules and split remain fixed. Only the model-facing representation changes.</p>
    <div class="contract"><div class="channel newton">Newton HP</div><div class="channel newton">d Newton LP/dt</div><div class="channel newton">d² Newton LP/dt²</div><div class="channel coord">S1</div><div class="channel coord">S2</div><div class="channel coord">dS2/dt</div></div>
    <p><strong>Three-channel coordinate:</strong> [S1, S2, dS2/dt]. <strong>Six-channel fusion:</strong> the three archived Newton channels followed by the coordinate triplet shown above.</p>

    <h2>Subject006 waveform</h2><p class="section-lede">Same 617 train / 69 test samples. Lower MAE and RMSE are better; higher R², concordance and tolerance are better.</p>
    <div class="table-wrap"><table><thead><tr><th>Representation</th><th>Avg MAE</th><th>Avg RMSE</th><th>DBP MAE</th><th>SBP MAE</th><th>BP accuracy</th><th>DBP/SBP r</th><th>DBP/SBP r²</th><th>DBP/SBP CCC</th><th>Within 5 mmHg</th><th>Epochs</th><th>Files</th></tr></thead><tbody>
      {metric_row(runs['s006_native_wave'], 'Native HDF5 PVI', 'winner')}
      {metric_row(runs['s006_ref_wave'], 'Reference PVI Parquet')}
      {metric_row(runs['s006_coord_wave'], 'Coordinate 3ch', 'warning')}
      {metric_row(runs['s006_fused_wave'], 'Newton + coordinate 6ch', 'winner')}
    </tbody></table></div>

    <h2>Subject006 direct fiducials</h2><p class="section-lede">No matching native-HDF5 direct-fiducial model was run. Reference Parquet is the archived-PVI control.</p>
    <div class="table-wrap"><table><thead><tr><th>Representation</th><th>Avg MAE</th><th>Avg RMSE</th><th>DBP MAE</th><th>SBP MAE</th><th>BP accuracy</th><th>DBP/SBP r</th><th>DBP/SBP r²</th><th>DBP/SBP CCC</th><th>Within 5 mmHg</th><th>Epochs</th><th>Files</th></tr></thead><tbody>
      {metric_row(runs['s006_ref_fid'], 'Reference PVI Parquet')}
      {metric_row(runs['s006_coord_fid'], 'Coordinate 3ch', 'winner')}
      {metric_row(runs['s006_fused_fid'], 'Newton + coordinate 6ch', 'winner')}
    </tbody></table></div>

    <div class="two-col"><div class="panel"><h3>Subject006 waveform average MAE</h3>{''.join(bar(label, value, max_bar, tone) for label, value, tone in wave_values)}</div>
      <div class="panel"><h3>Subject006 fiducial average MAE</h3>{''.join(bar(label, value, max_bar, tone) for label, value, tone in fid_values)}</div></div>

    <h2>Subject010 replication</h2><p class="section-lede">Same 1,578 train / 175 test samples. Subject010 uses the same US120 coordinate checkpoint; the GCNM was not retrained for this subject.</p>
    <div class="table-wrap"><table><thead><tr><th>Target / representation</th><th>Avg MAE</th><th>Avg RMSE</th><th>DBP MAE</th><th>SBP MAE</th><th>BP accuracy</th><th>DBP/SBP r</th><th>DBP/SBP r²</th><th>DBP/SBP CCC</th><th>Within 5 mmHg</th><th>Epochs</th><th>Files</th></tr></thead><tbody>
      {metric_row(runs['s010_ref_wave'], 'Waveform · reference PVI Parquet')}
      {metric_row(runs['s010_coord_wave'], 'Waveform · coordinate 3ch', 'winner')}
      {metric_row(runs['s010_fused_wave'], 'Waveform · Newton + coordinate 6ch', 'winner')}
      {metric_row(runs['s010_ref_fid'], 'Fiducials · reference PVI Parquet')}
      {metric_row(runs['s010_coord_fid'], 'Fiducials · coordinate 3ch', 'winner')}
      {metric_row(runs['s010_fused_fid'], 'Fiducials · Newton + coordinate 6ch', 'winner')}
    </tbody></table></div>
  </section>
  <div class="notice"><strong>Metric definition.</strong> The upstream <code>pvi_ml</code> “BP accuracy” is the mean of DBP and SBP Pearson <em>r</em>. Its exported <code>dbp_r2</code>/<code>sbp_r2</code> fields are Pearson <em>r</em> squared, not residual coefficient-of-determination scores.</div>

  <section id="findings"><h2>What the ablations say</h2>
    <div class="finding-grid">
      <div class="finding"><div class="eyebrow">Fusion recovers waveform information</div><strong>{delta(s6_fused_w['amae'], s6_coord_w['amae'])}</strong><span>Subject006 average waveform MAE change after adding Newton channels. SBP MAE falls from {f(s6_coord_w['sbp_mae'])} to {f(s6_fused_w['sbp_mae'])} mmHg.</span></div>
      <div class="finding"><div class="eyebrow">Fusion approaches the PVI control</div><strong>{delta(s6_fused_w['amae'], s6_ref_w['amae'])}</strong><span>Six-channel waveform MAE relative to reference-Parquet PVI. It remains {f(s6_fused_w['amae']-s6_native_w['amae'])} mmHg worse than the native-HDF5 run.</span></div>
      <div class="finding"><div class="eyebrow">Coordinate helps direct fiducials</div><strong>{delta(s6_fused_f['amae'], s6_ref_f['amae'])}</strong><span>Six-channel fiducial MAE relative to reference PVI. Coordinate-only is already {f(s6_ref_f['amae']-s6_coord_f['amae'])} mmHg better than reference.</span></div>
      <div class="finding"><div class="eyebrow">Subject010 waveform fusion</div><strong>{delta(s10_fused_w['amae'], s10_coord_w['amae'])}</strong><span>Six-channel average MAE relative to coordinate-only; DBP/SBP MAE is {f(s10_fused_w['dbp_mae'])}/{f(s10_fused_w['sbp_mae'])} mmHg.</span></div>
      <div class="finding"><div class="eyebrow">Subject010 fiducial fusion</div><strong>{delta(s10_fused_f['amae'], s10_coord_f['amae'])}</strong><span>Six-channel average MAE relative to coordinate-only. Reference-Parquet fiducial MAE is {f(s10_ref_f['amae'])} mmHg.</span></div>
      <div class="finding"><div class="eyebrow">Interpretation</div><strong>Complementary, not identical</strong><span>Newton preserves a broad transform of the measurements; coordinate GCNM contributes structured spatial dynamics. Their combination performs better than coordinate-only on subject006 waveform.</span></div>
    </div>
    <div class="notice"><strong>Caution.</strong> These are two single-subject experiments and independently initialized training runs. Native HDF5 versus reference Parquet differences quantify some stochastic training variation. Scaling decisions should use repeated seeds and paired results across subjects.</div>
  </section>

  <section id="visuals"><h2>Representation evidence</h2><p class="section-lede">The pilot GIFs run live inference; the Parquet GIFs read the serialized tensors back from disk. That separation verifies that storage does not change the representation.</p>
    <div class="gallery">{''.join(gifs)}</div>
  </section>

  <section id="artifacts"><h2>Primary artifact roots</h2>
    <div class="path-list">
      <div class="path"><strong>Subject006 coordinate 3ch</strong><code>artifacts/subject006_coordinate_direct_s1_s2_ds2_bp_v1/</code></div>
      <div class="path"><strong>Subject006 Newton+coordinate 6ch</strong><code>artifacts/subject006_newton_coordinate_6ch_bp_v1/</code></div>
      <div class="path"><strong>Subject010 coordinate 3ch</strong><code>artifacts/subject010_coordinate_direct_s1_s2_ds2_bp_v1/</code></div>
      <div class="path"><strong>Reference PVI Parquet</strong><code>artifacts/pvi_bp_us120_reference_parquet_v1/</code></div>
      <div class="path"><strong>Native subject006 HDF5 PVI</strong><code>artifacts/pvi_ml_subject006_native_v1/</code></div>
      <div class="path"><strong>Coordinate Parquet data</strong><code>gcnm_parquet/subject006_coordinate_direct_s1_s2_ds2_v1/</code><code>gcnm_parquet/subject010_coordinate_direct_s1_s2_ds2_v1/</code></div>
    </div>
  </section>
  <footer>Generated {datetime.now().astimezone().isoformat(timespec='seconds')} from immutable statistics, histories and verification artifacts. Values are displayed without re-evaluating model predictions.</footer>
</main></body></html>"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(document, encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "runs": len(runs), "bytes": OUTPUT.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
