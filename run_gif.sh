python3 - <<'PY'
from trainv2 import (
    make_spiral,
    random_projection,
    train_one,
    generate_vec_trajectory_2d,
    save_time_evolution_gif,
)

D =  512
param = "x"
n_points = 1000
train_steps = 8001
batch_size = 512
lr = 1e-3
noise_scale = 2.0
sample_steps = 200
frame_stride = 1
out_path = "flow_D2_x.gif"

gt_2d = make_spiral(n_points)
P = random_projection(D)
data_D = gt_2d @ P.T

model = train_one(
    data_D=data_D,
    D=D,
    param=param,
    steps=train_steps,
    batch_size=batch_size,
    lr=lr,
    noise_scale=noise_scale,
)

t_frames, frames_2d = generate_vec_trajectory_2d(
    model=model,
    P=P,
    D=D,
    param=param,
    n_samples=n_points,
    steps=sample_steps,
    noise_scale=noise_scale,
    frame_stride=frame_stride,
)

save_time_evolution_gif(
    t_frames=t_frames,
    frames_2d=frames_2d,
    out_path=out_path,
    gt_2d=gt_2d,
    title_prefix=f"D={D}, param={param}",
    fps=10,
    point_size=1,
    fixed_xy=True,
    hold_last_seconds=5.0,
)
PY
