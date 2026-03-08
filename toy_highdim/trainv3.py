# ============================================================
# Flow-Matching Toy Experiment (Buried 2D Spiral in D-dim)
# Reproduces a figure like: ground-truth / x-pred / eps-pred / v-pred
# AND also shows PCA scatter plots for:
#   - buried data (D->2)
#   - pure noise (D->2)
#   - mixed z_t (D->2)
#
# Run (recommended GPU selection):
#   CUDA_VISIBLE_DEVICES=0 python flow_matching_toy.py
# or in notebook: just run the whole cell.
# ============================================================

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # headless backend, no GUI needed
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.decomposition import PCA
from typing import Optional

def rho_prefix(rho: float):
    return f"rho{rho:.3g}".replace(".", "p").replace("-", "m")

def append_signal_stats(csv_path, txt_path, rho, D, gamma_eff, base_E, extra_E, d_eff, k95):
    added_E = (gamma_eff ** 2) * extra_E
    rho_actual = added_E / (base_E + 1e-12)

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if not file_exists:
            f.write("rho,D,gamma_eff,base_E,extra_E,added_E,rho_actual,d_eff,k95\n")
        f.write(
            f"{rho},{D},{gamma_eff},{base_E},{extra_E},{added_E},{rho_actual},{d_eff},{k95}\n"
        )

    txt_exists = os.path.exists(txt_path)
    with open(txt_path, "a") as f:
        if not txt_exists:
            f.write("rho\tD\tgamma_eff\tbase_E\textra_E\tadded_E\trho_actual\td_eff\tk95\n")
        f.write(
            f"{rho}\t{D}\t{gamma_eff}\t{base_E}\t{extra_E}\t{added_E}\t{rho_actual}\t{d_eff}\t{k95}\n"
        )



# -----------------------------
# Global settings
# -----------------------------
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# 1) Create 2D spiral data
# -----------------------------
def make_spiral(n=10_000):
    u = np.linspace(0, 1, n)
    theta = np.sqrt(u * (4 * np.pi) ** 2)
    r = theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    data = np.stack([x, y], axis=1)
    data /= np.std(data)
    return data.astype(np.float32)  # (N,2)

# -----------------------------
# 2) Fixed random column-orthogonal projection P (D x 2)
# -----------------------------
def random_projection(D: int):
    A = np.random.randn(D, 2)
    Q, _ = np.linalg.qr(A)          # Q has orthonormal columns
    return Q.astype(np.float32)     # (D,2)

def sample_t_b2b(B, device, P_mean=-0.8, P_std=0.8):
    t_z = torch.randn(B, device=device) * P_std + P_mean
    return torch.sigmoid(t_z)  # (B,) in (0,1)

# -----------------------------
# 3) Simple time-conditioned MLP
# -----------------------------
class MLP(nn.Module):
    def __init__(self, D, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, D),
        )

    def forward(self, x, t):
        # flow matching uses continuous t in [0,1]
        t = t.unsqueeze(1)  # (B,1)
        return self.net(torch.cat([x, t], dim=1))

# -----------------------------
# 4) Flow matching mixing + targets (matches your B2B style)
#    z_t = t*x + (1-t)*e,  e ~ N(0,I)
# -----------------------------
def mix_z_t(x, t, noise_scale=1.0):
    """
    x: (B,D)
    t: (B,) in [0,1]
    returns: z (B,D), e (B,D)
    """
    e = torch.randn_like(x) * noise_scale
    t_view = t.unsqueeze(1)
    z = t_view * x + (1.0 - t_view) * e
    return z, e

def velocity_from_model_output(z, out, t, param, t_eps=5e-2, t_min=1e-4):
    """
    Convert model output (x_hat / eps_hat / v_hat) into velocity v(z,t).

    Uses B2B-style clamp on (1-t): (1-t).clamp_min(t_eps)
    Uses a small clamp on t only for eps->x_hat conversion.
    """
    one_minus = (1.0 - t).clamp_min(t_eps).unsqueeze(1)
    t_view = t.unsqueeze(1)

    if param == "v":
        return out

    if param == "x":
        x_hat = out
        return (x_hat - z) / one_minus

    if param == "eps":
        eps_hat = out
        # For z = t x + (1-t) e, we have v = (z - e)/t
        t_safe = t.clamp_min(t_min).unsqueeze(1)
        return (z - eps_hat) / t_safe

    raise ValueError("param must be one of: x, eps, v")

def add_orthogonal_highdim_signal(
    data_D: np.ndarray,
    P: np.ndarray,
    gamma: Optional[float] = None,
    rho: Optional[float] = None,
    seed: int = 0,
):
    """
    Adds Gaussian signal in orthogonal complement of span(P).

    You can specify either:
      - gamma: direct scale
      - rho:   target energy ratio (recommended)

    rho is defined by:
      E||gamma * extra_orth||^2 = rho * E||data_D||^2
    """
    assert (gamma is None) ^ (rho is None), "Provide exactly one of gamma or rho."

    rng = np.random.default_rng(seed)
    N, D = data_D.shape

    extra = rng.standard_normal(size=(N, D)).astype(np.float32)
    extra_orth = extra - (extra @ P) @ P.T  # remove components in span(P)

    # compute per-sample mean squared norm
    base_E = float(np.mean(np.sum(data_D.astype(np.float64) ** 2, axis=1)))
    extra_E = float(np.mean(np.sum(extra_orth.astype(np.float64) ** 2, axis=1)))

    if rho is not None:
        if rho <= 0 or extra_E <= 1e-12:
            return data_D.astype(np.float32), 0.0, base_E, extra_E
        gamma = float(np.sqrt(rho * base_E / (extra_E + 1e-12)))

    out = (data_D + gamma * extra_orth).astype(np.float32)
    return out, float(gamma), base_E, extra_E



def pca_effective_dim(x_np: np.ndarray):
    """
    Returns:
      d_eff: participation-ratio effective dimension
      k95:   number of PCs to explain 95% variance
    """
    x = x_np.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)

    # SVD => cov eigenvalues are (S^2)/(N-1)
    _, S, _ = np.linalg.svd(x, full_matrices=False)
    lam = (S**2) / max(x.shape[0] - 1, 1)

    d_eff = (lam.sum() ** 2) / (np.square(lam).sum() + 1e-12)
    explained = lam / (lam.sum() + 1e-12)
    cum = np.cumsum(explained)
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    return float(d_eff), k95

def _join_prefix(*parts):
    clean = []
    for part in parts:
        if part is None:
            continue
        part = str(part).strip()
        if not part:
            continue
        clean.append(part.strip("_"))
    return "_".join(clean)


def prefixed_name(filename: str, *prefixes):
    prefix = _join_prefix(*prefixes)
    if not prefix:
        return filename
    return f"{prefix}_{filename}"


def prefixed_path(path: str, *prefixes):
    dir_name, base_name = os.path.split(path)
    new_base_name = prefixed_name(base_name, *prefixes)
    if dir_name:
        return os.path.join(dir_name, new_base_name)
    return new_base_name


def plot_rho_vs_gamma(csv_path: str, out_path: str):
    if not os.path.exists(csv_path):
        print(f"Skip rho-vs-gamma plot: missing stats file {csv_path}")
        return

    gamma_by_D = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            D = int(float(row["D"]))
            rho = float(row["rho"])
            gamma_eff = float(row["gamma_eff"])
            if D not in gamma_by_D:
                gamma_by_D[D] = {}
            gamma_by_D[D][rho] = gamma_eff

    if not gamma_by_D:
        print(f"Skip rho-vs-gamma plot: no rows in {csv_path}")
        return

    out_parent = os.path.dirname(out_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for D in sorted(gamma_by_D):
        pairs = sorted(gamma_by_D[D].items(), key=lambda x: x[0])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=4, label=f"D={D}")

    if any((x > 0.0) for xs in gamma_by_D.values() for x in xs):
        ax.set_xscale("symlog", linthresh=1e-2)
    ax.set_xlabel("rho")
    ax.set_ylabel("gamma_eff")
    ax.set_title("Gamma vs Rho")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")

# -----------------------------
# 5) PCA helper (for visualization only)
# -----------------------------
def pca2(x_nd: np.ndarray):
    return PCA(n_components=2).fit_transform(x_nd)

# -----------------------------
# 6) Visualize generation pipeline with PCA (for each D)
# -----------------------------
def _square_limits_from_arrays(arrays, pad=0.08):
    """
    Build fixed square x/y limits from a list of (N,2) arrays.
    """
    all_xy = np.concatenate(arrays, axis=0)
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half = 0.5 * max(x_max - x_min, y_max - y_min) * (1.0 + pad)
    return (cx - half, cx + half), (cy - half, cy + half)


def plot_generation_pca(
    gt_2d,
    Ds,
    t_mix=0.5,
    t_mix_near1=0.95,
    noise_scale=1.0,
    n_show=8000,
    fixed_xy=True,
    xy_limits=None,
    rho=0.0, 
    gamma_seed=0,
    out_prefix="",
    out_dir=".",
):
    """
    Row per D:
      [ground truth 2D] [PCA(buried data)] [PCA(noise)] [PCA(mixed z_t, t=...)] ...
    """
    N = min(n_show, gt_2d.shape[0])
    gt = gt_2d[:N]

    mix_times = [float(t_mix)]
    if t_mix_near1 is not None:
        t_near = float(t_mix_near1)
        if abs(t_near - mix_times[0]) > 1e-8:
            mix_times.append(t_near)

    n_cols = 3 + len(mix_times)
    fig, axes = plt.subplots(len(Ds), n_cols, figsize=(3.5 * n_cols, 3.5 * len(Ds)))
    if len(Ds) == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Ground-truth (2D)",
        "PCA(buried data D→2)",
        "PCA(noise e D→2)",
    ]
    col_titles += [f"PCA(mixed z_t D→2), t={tm:.2f}" for tm in mix_times]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title)

    rows = []
    for D in Ds:
        P = random_projection(D)
        data_D = gt @ P.T
        data_D, _, _, _ = add_orthogonal_highdim_signal(data_D, P, rho=rho, seed=gamma_seed)


        x = torch.from_numpy(data_D).to(device)
        e = torch.randn_like(x) * noise_scale

        data_pca = pca2(data_D)
        e_pca = pca2(e.detach().cpu().numpy())
        z_pcas = []
        for tm in mix_times:
            z = tm * x + (1.0 - tm) * e
            z_pcas.append(pca2(z.detach().cpu().numpy()))
        rows.append((D, data_pca, e_pca, z_pcas))

    if fixed_xy:
        if xy_limits is None:
            x_lim, y_lim = _square_limits_from_arrays(
                [gt] + [arr for _, data_pca, e_pca, z_pcas in rows for arr in ([data_pca, e_pca] + z_pcas)]
            )
        else:
            x_lim, y_lim = xy_limits

    for i, (D, data_pca, e_pca, z_pcas) in enumerate(rows):

        axes[i, 0].scatter(gt[:, 0], gt[:, 1], s=1)
        axes[i, 0].set_ylabel(f"D={D}", rotation=0, labelpad=25, va="center")
        axes[i, 0].set_aspect("equal", "box")

        axes[i, 1].scatter(data_pca[:, 0], data_pca[:, 1], s=1)
        axes[i, 1].set_aspect("equal", "box")

        axes[i, 2].scatter(e_pca[:, 0], e_pca[:, 1], s=1)
        axes[i, 2].set_aspect("equal", "box")

        for k, z_pca in enumerate(z_pcas):
            col = 3 + k
            axes[i, col].scatter(z_pca[:, 0], z_pca[:, 1], s=1)
            axes[i, col].set_aspect("equal", "box")

        for j in range(n_cols):
            axes[i, j].tick_params(labelsize=7)
            if fixed_xy:
                axes[i, j].set_xlim(x_lim)
                axes[i, j].set_ylim(y_lim)


    plt.tight_layout()
    prefix = rho_prefix(rho)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, prefixed_name("fig_generation_pca.png", out_prefix, prefix))
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")

def plot_generation_with_projection_matrix(
    gt_2d,
    Ds,
    t_mix=0.5,
    t_mix_near1=0.95,
    noise_scale=1.0,
    n_show=8000,
    fixed_xy=True,
    xy_limits=None,
    rho=0.0, 
    gamma_seed=0,
    out_prefix="",
    out_dir=".",
):
    """
    Row per D:
      [ground truth 2D] [x_D @ P] [e_D @ P] [z_t @ P, t=...] ...

    Uses the known projection matrix P to map D-dim vectors back to 2D.
    """
    N = min(n_show, gt_2d.shape[0])
    gt = gt_2d[:N]

    mix_times = [float(t_mix)]
    if t_mix_near1 is not None:
        t_near = float(t_mix_near1)
        if abs(t_near - mix_times[0]) > 1e-8:
            mix_times.append(t_near)

    n_cols = 3 + len(mix_times)
    fig, axes = plt.subplots(len(Ds), n_cols, figsize=(3.5 * n_cols, 3.5 * len(Ds)))
    if len(Ds) == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Ground-truth (2D)",
        "Back-proj(buried) x_D @ P",
        "Back-proj(noise) e_D @ P",
    ]
    col_titles += [f"Back-proj(mixed) z_t @ P, t={tm:.2f}" for tm in mix_times]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title)

    rows = []
    for D in Ds:

        P = random_projection(D)
        data_D = gt @ P.T
        data_D, _, _, _ = add_orthogonal_highdim_signal(data_D, P, rho=rho, seed=gamma_seed)

        x = torch.from_numpy(data_D).to(device)
        e = torch.randn_like(x) * noise_scale

        data_back = data_D @ P
        e_back = e.detach().cpu().numpy() @ P
        z_backs = []
        for tm in mix_times:
            z = tm * x + (1.0 - tm) * e
            z_backs.append(z.detach().cpu().numpy() @ P)
        rows.append((D, data_back, e_back, z_backs))

    if fixed_xy:
        if xy_limits is None:
            x_lim, y_lim = _square_limits_from_arrays(
                [gt] + [arr for _, data_back, e_back, z_backs in rows for arr in ([data_back, e_back] + z_backs)]
            )
        else:
            x_lim, y_lim = xy_limits

    for i, (D, data_back, e_back, z_backs) in enumerate(rows):

        axes[i, 0].scatter(gt[:, 0], gt[:, 1], s=1)
        axes[i, 0].set_ylabel(f"D={D}", rotation=0, labelpad=25, va="center")
        axes[i, 0].set_aspect("equal", "box")

        axes[i, 1].scatter(data_back[:, 0], data_back[:, 1], s=1)
        axes[i, 1].set_aspect("equal", "box")

        axes[i, 2].scatter(e_back[:, 0], e_back[:, 1], s=1)
        axes[i, 2].set_aspect("equal", "box")

        for k, z_back in enumerate(z_backs):
            col = 3 + k
            axes[i, col].scatter(z_back[:, 0], z_back[:, 1], s=1)
            axes[i, col].set_aspect("equal", "box")

        for j in range(n_cols):
            axes[i, j].tick_params(labelsize=7)
            if fixed_xy:
                axes[i, j].set_xlim(x_lim)
                axes[i, j].set_ylim(y_lim)

    plt.tight_layout()
    prefix = rho_prefix(rho)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, prefixed_name("fig_generation_projection_matrix.png", out_prefix, prefix))
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    plt.close(fig)


# -----------------------------
# 7) Train one model (x / eps / v) for a given D
# -----------------------------

def train_one(data_D: np.ndarray, D: int, param: str,
              steps=4000, batch_size=512, lr=1e-3, noise_scale=1.0,
              t_eps=5e-2, P_mean=-0.8, P_std=0.8):
    model = MLP(D).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x_all = torch.from_numpy(data_D).to(device)
    N = x_all.shape[0]

    for step in range(steps):
        idx = torch.randint(0, N, (batch_size,), device=device)
        x = x_all[idx]  # (B,D)

        # B2B-like t sampling
        t = sample_t_b2b(batch_size, device=device, P_mean=P_mean, P_std=P_std)  # (B,)

        # Flow matching mix
        z, e = mix_z_t(x, t, noise_scale=noise_scale)

        # B2B target velocity
        denom = (1.0 - t).clamp_min(t_eps).unsqueeze(1)
        v_target = (x - z) / denom

        # Model output type depends on param
        out = model(z, t)

        # Convert output to velocity prediction
        v_pred = velocity_from_model_output(z, out, t, param, t_eps=t_eps)

        # ALWAYS velocity loss (B2B-style)
        loss = loss_fn(v_pred, v_target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 1000 == 0:
            print(f"[train-vloss] D={D:>3} param={param:>3} step={step:>5} loss={loss.item():.4f}")

    return model

# ============================================================
# 8) Sample generation (spiral toy, vector z in R^D, NO CFG)
# Euler or Heun integration of dz/dt = v_theta(z,t)
# ============================================================

@torch.no_grad()
def _forward_sample_vec(model, z, t, param, t_eps=5e-2, t_min=1e-4):
    """
    z: (B, D)
    t: (B,) in [0,1]
    returns v_pred: (B, D)
    """
    out = model(z, t)  # (B, D) output depends on param
    v_pred = velocity_from_model_output(
        z=z,
        out=out,
        t=t,
        param=param,
        t_eps=t_eps,
        t_min=t_min,
    )
    return v_pred


@torch.no_grad()
def euler_step_vec(model, z, t, t_next, param, t_eps=5e-2, t_min=1e-4):
    """
    Euler: z_next = z + (t_next - t) * v(z,t)
    """
    dt = (t_next - t).unsqueeze(1)  # (B,1)
    v = _forward_sample_vec(model, z, t, param, t_eps=t_eps, t_min=t_min)
    return z + dt * v


@torch.no_grad()
def heun_step_vec(model, z, t, t_next, param, t_eps=5e-2, t_min=1e-4):
    """
    Heun (predictor-corrector):
      v_t = v(z,t)
      z_euler = z + dt*v_t
      v_tnext = v(z_euler,t_next)
      z_next = z + dt*0.5*(v_t + v_tnext)
    """
    dt = (t_next - t).unsqueeze(1)  # (B,1)

    v_t = _forward_sample_vec(model, z, t, param, t_eps=t_eps, t_min=t_min)
    z_euler = z + dt * v_t

    v_tnext = _forward_sample_vec(model, z_euler, t_next, param, t_eps=t_eps, t_min=t_min)

    v = 0.5 * (v_t + v_tnext)
    return z + dt * v


@torch.no_grad()
def generate_vec(model, D, param,
                 n_samples=10_000,
                 steps=50,
                 noise_scale=1.0,
                 method="heun",
                 t_eps=5e-2,
                 t_min=1e-4,
                 z0=None):
    """
    Spiral-toy sampler (vector version).
    Starts at z(0) ~ N(0, noise_scale^2 I), integrates to t=1.

    Args:
        model: trained MLP
        D: dimension
        param: "x" | "eps" | "v"
        n_samples: number of samples to generate
        steps: number of integration steps (50 matches B2B default)
        noise_scale: initial noise std
        method: "euler" or "heun"
        t_eps: clamp for (1-t) denominators (B2B uses 5e-2)
        t_min: clamp for t denominators (eps conversion)
    """
    device = next(model.parameters()).device

    # initial noise
    if z0 is None:
        z = noise_scale * torch.randn(n_samples, D, device=device)
    else:
        z = z0.to(device).clone()
        n_samples = z.shape[0]

    # time grid
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)

    # choose integrator
    if method == "euler":
        stepper = euler_step_vec
    elif method == "heun":
        stepper = heun_step_vec
    else:
        raise NotImplementedError(f"Unknown method: {method}")

    # integrate
    for i in range(steps - 1):
        t_val = ts[i].item()
        t_next_val = ts[i + 1].item()

        t = torch.full((n_samples,), t_val, device=device)
        t_next = torch.full((n_samples,), t_next_val, device=device)

        z = stepper(model, z, t, t_next, param, t_eps=t_eps, t_min=t_min)

    # last step: Euler (like B2B)
    t = torch.full((n_samples,), ts[-2].item(), device=device)
    t_next = torch.full((n_samples,), ts[-1].item(), device=device)
    z = euler_step_vec(model, z, t, t_next, param, t_eps=t_eps, t_min=t_min)

    return z


@torch.no_grad()
def generate_vec_trajectory_2d(
    model,
    P,
    D,
    param,
    n_samples=10_000,
    steps=50,
    noise_scale=1.0,
    method="heun",
    t_eps=5e-2,
    t_min=1e-4,
    frame_stride=1,
    z0=None,
):
    """
    Generate and store 2D back-projected snapshots across time.
    Returns:
      t_frames: np.ndarray [K]
      frames_2d: list[np.ndarray], each (n_samples, 2)
    """
    device = next(model.parameters()).device
    P_t = torch.from_numpy(P).to(device)

    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")

    if z0 is None:
        z = noise_scale * torch.randn(n_samples, D, device=device)
    else:
        z = z0.to(device).clone()
        n_samples = z.shape[0]
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)

    if method == "euler":
        stepper = euler_step_vec
    elif method == "heun":
        stepper = heun_step_vec
    else:
        raise NotImplementedError(f"Unknown method: {method}")

    t_frames = [0.0]
    frames_2d = [(z @ P_t).detach().cpu().numpy()]

    for i in range(steps - 1):
        t_val = ts[i].item()
        t_next_val = ts[i + 1].item()

        t = torch.full((n_samples,), t_val, device=device)
        t_next = torch.full((n_samples,), t_next_val, device=device)
        z = stepper(model, z, t, t_next, param, t_eps=t_eps, t_min=t_min)

        if ((i + 1) % frame_stride) == 0:
            t_frames.append(t_next_val)
            frames_2d.append((z @ P_t).detach().cpu().numpy())

    # last step: Euler
    t = torch.full((n_samples,), ts[-2].item(), device=device)
    t_next = torch.full((n_samples,), ts[-1].item(), device=device)
    z = euler_step_vec(model, z, t, t_next, param, t_eps=t_eps, t_min=t_min)

    if len(t_frames) == 0 or t_frames[-1] < 1.0:
        t_frames.append(1.0)
        frames_2d.append((z @ P_t).detach().cpu().numpy())

    return np.array(t_frames, dtype=np.float32), frames_2d


def save_time_evolution_gif(
    t_frames,
    frames_2d,
    out_path="fig_time_evolution.gif",
    gt_2d=None,
    title_prefix="Time evolution",
    fps=10,
    point_size=1,
    fixed_xy=True,
    xy_limits=None,
    hold_last_seconds=2.0,
):
    """
    Save a GIF that shows sample evolution over time.
    """
    out_parent = os.path.dirname(out_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    if fixed_xy:
        if xy_limits is None:
            arrays = list(frames_2d)
            if gt_2d is not None:
                arrays = [gt_2d] + arrays
            x_lim, y_lim = _square_limits_from_arrays(arrays)
        else:
            x_lim, y_lim = xy_limits
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
    ax.set_aspect("equal", "box")

    if gt_2d is not None:
        ax.scatter(gt_2d[:, 0], gt_2d[:, 1], s=point_size, c="lightgray", alpha=0.25, label="gt")

    hold_last_frames = max(0, int(round(float(hold_last_seconds) * max(fps, 1))))
    frame_ids = list(range(len(frames_2d)))
    if hold_last_frames > 0 and len(frames_2d) > 0:
        frame_ids.extend([len(frames_2d) - 1] * hold_last_frames)

    scat = ax.scatter(frames_2d[frame_ids[0]][:, 0], frames_2d[frame_ids[0]][:, 1], s=point_size, c="tab:blue")
    title = ax.set_title(f"{title_prefix}, t={t_frames[frame_ids[0]]:.2f}")
    ax.tick_params(labelsize=8)

    def _update(k):
        fid = frame_ids[k]
        scat.set_offsets(frames_2d[fid])
        title.set_text(f"{title_prefix}, t={t_frames[fid]:.2f}")
        return scat, title

    anim = animation.FuncAnimation(
        fig, _update, frames=len(frame_ids), interval=int(1000 / max(fps, 1)), blit=False, repeat=False
    )
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved {out_path}")


# -----------------------------
# 9) Full experiment: D rows, [GT, x-pred, eps-pred, v-pred]
# -----------------------------
def plot_full_experiment_flow_matching(
    #Ds=(2, 8, 16, 512),
    Ds=(2, 8),
    n_points=100,
    train_steps_map=None,
    batch_size=512,
    lr=1e-3,
    noise_scale=1.0,
    sample_steps=50,
    fixed_xy=True,
    xy_limits=None,
    return_artifacts=False,
    rho=0.0,
    gamma=None,             # backward-compatible alias; if set, overrides rho
    gamma_seed=0,
    out_prefix="",
    out_dir=".",
):
    if gamma is not None:
        rho = float(gamma)

    if train_steps_map is None:
        # you can increase these for cleaner plots (especially D=512)
        #train_steps_map = {2: 2000, 8: 3000, 16: 4000, 512: 6000}
        train_steps_map = {2: 2001, 8: 3001}

    gt_2d = make_spiral(n_points)
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(len(Ds), 4, figsize=(12, 12))
    if len(Ds) == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["ground-truth", "x-pred", "ε-pred", "v-pred"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title)

    all_2d = [gt_2d]
    artifacts = {}
    for i, D in enumerate(Ds):
        # fixed projection for this D (buries 2D data into D dims)
        P = random_projection(D)        # (D,2)
        data_D = gt_2d @ P.T            # (N,D)


        data_D, gamma_eff, base_E, extra_E = add_orthogonal_highdim_signal( data_D, P, rho=rho, seed=gamma_seed  )
        d_eff, k95 = pca_effective_dim(data_D)

        added_E = (gamma_eff ** 2) * extra_E
        rho_actual = added_E / (base_E + 1e-12)

        print(
            f"[data stats] D={D} rho={rho:.3g} gamma_eff={gamma_eff:.6g} "
            f"base_E={base_E:.6g} extra_E={extra_E:.6g} "
            f"added_E={added_E:.6g} rho_actual={rho_actual:.6g} "
            f"d_eff={d_eff:.1f} k95={k95}"
        )

        csv_path = os.path.join(out_dir, "all_signal_stats.csv")
        txt_path = os.path.join(out_dir, "all_signal_stats.txt")
        append_signal_stats(csv_path, txt_path, rho, D, gamma_eff, base_E, extra_E, d_eff, k95)


        # GT plot
        ax = axes[i, 0]
        ax.scatter(gt_2d[:, 0], gt_2d[:, 1], s=1)
        ax.set_ylabel(f"D={D}", rotation=0, labelpad=25, va="center")
        ax.set_aspect("equal", "box")

        # train 3 models
        models = {}
        for param in ["x", "eps", "v"]:
            steps = train_steps_map.get(D, 4000)
            models[param] = train_one(
                data_D=data_D,
                D=D,
                param=param,
                steps=steps,
                batch_size=batch_size,
                lr=lr,
                noise_scale=noise_scale,
            )
        artifacts[D] = {"P": P, "models": models, "z0_by_param": {}}

        # sample & plot each
        for j, param in enumerate(["x", "eps", "v"], start=1):
            z0 = noise_scale * torch.randn(n_points, D, device=device)
            xD = generate_vec(
                models[param],
                D=D,
                param=param,
                n_samples=n_points,
                steps=sample_steps,
                noise_scale=noise_scale,
                method="heun",     # or "euler"
                t_eps=5e-2,
                t_min=1e-4,
                z0=z0,
            )
            x2d = (xD.detach().cpu().numpy()) @ P  # back to 2D for visualization
            all_2d.append(x2d)
            artifacts[D]["z0_by_param"][param] = z0.detach().cpu()

            ax = axes[i, j]
            ax.scatter(x2d[:, 0], x2d[:, 1], s=1)
            ax.set_aspect("equal", "box")

        for j in range(4):
            axes[i, j].tick_params(labelsize=7)

    if fixed_xy:
        if xy_limits is None:
            x_lim, y_lim = _square_limits_from_arrays(all_2d)
        else:
            x_lim, y_lim = xy_limits
        for i in range(len(Ds)):
            for j in range(4):
                axes[i, j].set_xlim(x_lim)
                axes[i, j].set_ylim(y_lim)

    plt.tight_layout()
    signal_prefix = rho_prefix(rho)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, prefixed_name("fig_flow_matching.png", out_prefix, signal_prefix))
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    if return_artifacts:
        return gt_2d, artifacts
    return gt_2d

def run_all_results_single_process(
    Ds=(2, 8),
    n_points=1000,
    n_show=None,
    train_steps_map=None,
    batch_size=512,
    lr=1e-4,
    noise_scale=1.0,
    sample_steps=50,
    flow_xy_limits=((-2, 3), (-2, 2)),
    gif_D=2,
    gif_param="x",
    gif_sample_steps=None,
    gif_frame_stride=1,
    gif_out_path="flow_D2_x.gif",
    gif_hold_last_seconds=2.0,
    make_gif=True,
    rho=0.0,
    gamma=None,  # backward-compatible alias; if set, overrides rho
    gamma_seed=0, 
    out_prefix="",
    out_dir="outputs",
):
    """
    One-process pipeline:
      1) PCA/projection visualization
      2) full train+sample figure
      3) optional time-evolution GIF from one trained model

    Outputs are saved under `out_dir` (default: `outputs/`).
    """
    if gamma is not None:
        rho = float(gamma)

    signal_prefix = rho_prefix(rho)
    os.makedirs(out_dir, exist_ok=True)
    if train_steps_map is None:
        train_steps_map = {2: 8001, 8: 16001}
    if n_show is None:
        # Keep one sample-count knob by default.
        n_show = n_points
    if gif_sample_steps is None:
        # Keep GIF solver steps aligned with static generation by default.
        gif_sample_steps = sample_steps

    gt_2d_vis = make_spiral(n_points)
    plot_generation_pca(
        gt_2d=gt_2d_vis,
        Ds=Ds,
        t_mix=0.5,
        noise_scale=noise_scale,
        n_show=n_show,
        rho=rho,
        gamma_seed=gamma_seed,
        out_prefix=out_prefix,
        out_dir=out_dir,
    )
    plot_generation_with_projection_matrix(
        gt_2d=gt_2d_vis,
        Ds=Ds,
        t_mix=0.5,
        noise_scale=noise_scale,
        n_show=n_show,
        rho=rho,
        gamma_seed=gamma_seed,
        out_prefix=out_prefix,
        out_dir=out_dir,
    )

    gt_2d_flow, artifacts = plot_full_experiment_flow_matching(
        Ds=Ds,
        n_points=n_points,
        train_steps_map=train_steps_map,
        batch_size=batch_size,
        lr=lr,
        noise_scale=noise_scale,
        sample_steps=sample_steps,
        fixed_xy=True,
        xy_limits=flow_xy_limits,
        rho=rho,
        gamma_seed=gamma_seed,
        out_prefix=out_prefix,
        out_dir=out_dir,
        return_artifacts=True,
    )

    if not make_gif:
        return

    if gif_D not in artifacts:
        raise ValueError(f"gif_D={gif_D} not found in trained Ds={Ds}")
    if gif_param not in artifacts[gif_D]["models"]:
        raise ValueError(f"gif_param={gif_param} must be one of ['x', 'eps', 'v']")

    model = artifacts[gif_D]["models"][gif_param]
    P = artifacts[gif_D]["P"]
    z0 = artifacts[gif_D]["z0_by_param"][gif_param]
    t_frames, frames_2d = generate_vec_trajectory_2d(
        model=model,
        P=P,
        D=gif_D,
        param=gif_param,
        n_samples=n_points,
        steps=gif_sample_steps,
        noise_scale=noise_scale,
        method="heun",
        frame_stride=gif_frame_stride,
        z0=z0,
    )
    save_time_evolution_gif(
        t_frames=t_frames,
        frames_2d=frames_2d,
        out_path=os.path.join(out_dir, prefixed_path(gif_out_path, out_prefix, signal_prefix)),
        gt_2d=gt_2d_flow,
        title_prefix=f"D={gif_D}, param={gif_param}",
        fps=10,
        point_size=1,
        fixed_xy=True,
        xy_limits=flow_xy_limits,
        hold_last_seconds=gif_hold_last_seconds,
    )

# ============================================================
# RUN EVERYTHING
# ============================================================

if __name__ == "__main__":
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    stats_csv_path = os.path.join(out_dir, "all_signal_stats.csv")
    stats_txt_path = os.path.join(out_dir, "all_signal_stats.txt")
    for path in (stats_csv_path, stats_txt_path):
        if os.path.exists(path):
            os.remove(path)

    rhos = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100, 500, 1000,2000]

    for rho in rhos:
        print("\n" + "="*80)
        print(f"Running rho={rho}")
        print("="*80)

        run_all_results_single_process(
            Ds=(2, 8, 16, 512),
            n_points=2000,
            train_steps_map={2: 8001, 8: 16001, 16: 24001, 512: 40001},
            batch_size=512,
            lr=5e-4,
            noise_scale=1.0,
            sample_steps=200,
            flow_xy_limits=((-2, 3), (-2, 2)),
            make_gif=False,
            rho=rho,
            gamma_seed=0,
            out_dir=out_dir,
        )

    plot_rho_vs_gamma(
        csv_path=stats_csv_path,
        out_path=os.path.join(out_dir, "rho_vs_gamma.png"),
    )
