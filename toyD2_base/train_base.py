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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # headless backend, no GUI needed
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.decomposition import PCA

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
        P = random_projection(D)        # (D,2)
        data_D = gt @ P.T               # (N,D)

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
    plt.savefig("fig_generation_pca.png", dpi=200)
    plt.close(fig)
    print("Saved fig_generation_pca.png")

def plot_generation_with_projection_matrix(
    gt_2d,
    Ds,
    t_mix=0.5,
    t_mix_near1=0.95,
    noise_scale=1.0,
    n_show=8000,
    fixed_xy=True,
    xy_limits=None,
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
        P = random_projection(D)    # (D,2)
        data_D = gt @ P.T           # (N,D)

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
    plt.savefig("fig_generation_projection_matrix.png", dpi=200)
    plt.close(fig)
    print("Saved fig_generation_projection_matrix.png")


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
):
    if train_steps_map is None:
        # you can increase these for cleaner plots (especially D=512)
        #train_steps_map = {2: 2000, 8: 3000, 16: 4000, 512: 6000}
        train_steps_map = {2: 2001, 8: 3001}

    gt_2d = make_spiral(n_points)

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
    plt.savefig("fig_flow_matching.png", dpi=200)
    plt.close(fig)
    print("Saved fig_flow_matching.png")
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
):
    """
    One-process pipeline:
      1) PCA/projection visualization
      2) full train+sample figure
      3) optional time-evolution GIF from one trained model
    """
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
    )
    plot_generation_with_projection_matrix(
        gt_2d=gt_2d_vis,
        Ds=Ds,
        t_mix=0.5,
        noise_scale=noise_scale,
        n_show=n_show,
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
        out_path=gif_out_path,
        gt_2d=gt_2d_flow,
        title_prefix=f"D={gif_D}, pred={gif_param}",
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
    run_all_results_single_process(
        Ds=(2, 8,16, 512),
        n_points=3000,
        train_steps_map={2: 8001, 8: 16001, 16:24001, 512:40001},
        batch_size=512,
        lr=5e-4,
        noise_scale=1.0,
        sample_steps=20,
        flow_xy_limits=((-2, 3), (-2, 2)),
        gif_D=16,
        gif_param="x",
        gif_sample_steps=None,
        gif_frame_stride=1,
        gif_out_path="flow_D16_x.gif",
        gif_hold_last_seconds=5.0,
        make_gif=True,
    )
