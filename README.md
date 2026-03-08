# Manifold Learning: Flow-Matching Toy Reproduction (2D Spiral in D-Dim)

This repo reproduces a toy flow-matching result:
- ground-truth 2D spiral
- embed to higher dimension `D`
- train `x` / `eps` / `v` parameterizations
- sample back to 2D and compare quality
- visualize time evolution as a GIF (Gaussian -> spiral)

<details>
<summary>Show usage, outputs, and config details</summary>

## Files

- `toyD2_base/train_base.py`: main code (data, training, sampling, plotting, GIF)
- `toyD2_base/run.sh`: one-process full pipeline (all figures + GIF)
- `toyD2_base/run_gif.sh`: GIF-only script (separate simple entrypoint)

## Requirements

Python 3.9+ recommended.

Install dependencies:

```bash
pip install numpy torch matplotlib scikit-learn pillow
```

Notes:
- GPU is optional. Code falls back to CPU.
- `matplotlib` uses `Agg` backend (headless, no GUI required).

## Quick Start

Run full pipeline:

```bash
bash toyD2_base/run.sh
```

Run GIF-only script:

```bash
bash toyD2_base/run_gif.sh
```

## Outputs

Running `toyD2_base/run.sh` produces:

`toyD2_base/fig_generation_pca.png`: PCA-based visualization of ground truth, buried data, noise, and mixed `z_t` states for each `D`.
<a href="./toyD2_base/fig_generation_pca.png"><img src="./toyD2_base/fig_generation_pca.png" alt=""></a>

`toyD2_base/fig_generation_projection_matrix.png`: Generation results projected back to 2D with the known projection matrix, comparing model parameterizations.
<a href="./toyD2_base/fig_generation_projection_matrix.png"><img src="./toyD2_base/fig_generation_projection_matrix.png" alt=""></a>

`toyD2_base/fig_flow_matching.png`: Main flow-matching comparison figure across dimensions and parameterizations.
<a href="./toyD2_base/fig_flow_matching.png"><img src="./toyD2_base/fig_flow_matching.png" alt=""></a>

`toyD2_base/flow_D16_x.gif`: Time-evolution animation from Gaussian noise to the spiral for `D=16`, `param=x`.
<a href="./toyD2_base/flow_D16_x.gif"><img src="./toyD2_base/flow_D16_x.gif" alt=""></a>

## Main Config Knobs

In `toyD2_base/train_base.py` inside `run_all_results_single_process(...)`:

- `n_points`: single sample-count knob (data + sampling + GIF points)
- `train_steps_map`: training steps per `D`
- `batch_size`, `lr`
- `sample_steps`: ODE integration steps for static generation figure
- `gif_sample_steps`: ODE steps for GIF (`None` means use `sample_steps`)
- `gif_D`, `gif_param`: choose which trained panel to animate
- `gif_frame_stride`: save one GIF frame every N solver steps
- `gif_hold_last_seconds`: keep final frame static at `t=1`

## Reproducibility

- Seed is fixed in code (`SEED = 0`) for NumPy and PyTorch.
- Static figure and GIF can share the same initial noise for the chosen panel, so final frame matches static result better.

</details>
