 # JiT Toy Reproduction (2D Spiral in D-Dim)

This repository reproduces a toy flow-matching result from Table 1 and Figure 2 of the paper “Back to Basics: Let Denoising Generative Models Denoise” 
- ground-truth 2D spiral
- embed to higher dimension `D`
- train `x` / `eps` / `v` parameterizations
- sample back to 2D and compare quality
- visualize time evolution as a GIF (Gaussian -> spiral)

## Files

- `trainv2.py`: main code (data, training, sampling, plotting, GIF)
- `run.sh`: one-process full pipeline (all figures + GIF)
- `run_gif.sh`: GIF-only script (separate simple entrypoint)

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
bash run.sh
```

Run GIF-only script:

```bash
bash run_gif.sh
```

## Outputs

Running `run.sh` produces:

- `fig_generation_pca.png`
- fig_generation_pca.png
- `fig_generation_projection_matrix.png`
- fig_generation_projection_matrix.png
- `fig_flow_matching.png`
- fig_flow_matching.png
- `flow_D2_x.gif`
- flow_D2_x.gif

## Main Config Knobs

In `trainv2.py` inside `run_all_results_single_process(...)`:

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
