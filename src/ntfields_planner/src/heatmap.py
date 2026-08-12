import numpy as np
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
from pyoctomap import OcTree
from scipy.spatial import cKDTree

mx.disable_compile()

OCTREE_PATH = "/Users/tarunjaikumar/Documents/Robotics-UCSD/ERL-Racecar/catkin_ws/src/mushr_gazebo/exports/octree.bt"
FREE_PATH   = "/Users/tarunjaikumar/Documents/Robotics-UCSD/ERL-Racecar/catkin_ws/src/mushr_gazebo/exports/free.txt"
RESOLUTION  = 0.05
GRID_N      = 200      # heatmap resolution per axis
MASK_RADIUS = 0.5      # a grid point counts as "mapped" if free space is this close

# these must match training.py
CONSTANT_SPEED = 1.0
DMIN, DMAX = 0.2, 5.5


class NTField(nn.Module):
    """Architecture must match training.py exactly."""

    def __init__(self):
        super().__init__()
        self.f_encoder = nn.Sequential(
            nn.Linear(2, 128), nn.Softplus(),
            nn.Linear(128, 128), nn.Softplus(),
        )
        self.tao_model = nn.Sequential(
            nn.Linear(256, 128), nn.Softplus(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def symmetric_operator(self, a, b):
        return mx.concatenate([mx.maximum(a, b), mx.minimum(a, b)], axis=-1)

    def tau(self, qs, qg):
        return self.tao_model(
            self.symmetric_operator(self.f_encoder(qs), self.f_encoder(qg))
        )

    def compute_T(self, qs, qg):
        r = mx.sqrt(mx.sum((qg - qs) ** 2, axis=-1, keepdims=True) + 1e-12)
        return r / self.tau(qs, qg)


def load_occupied(path, resolution):
    """Expand occupied octree leaves into a dense point cloud."""
    tree = OcTree(resolution)
    tree.readBinary(path)

    chunks = []
    for leaf in tree.begin_leafs():
        if not tree.isNodeOccupied(leaf):
            continue
        c = leaf.getCoordinate()
        size = leaf.getSize()
        n = int(size / resolution) + 1
        x = np.linspace(c[0] - size / 2, c[0] + size / 2, n)
        y = np.linspace(c[1] - size / 2, c[1] + size / 2, n)
        X, Y = np.meshgrid(x, y)
        chunks.append(np.stack([X.ravel(), Y.ravel()], axis=1))

    return np.unique(np.concatenate(chunks, axis=0), axis=0)


def speed_star(xy, kdtree):
    d, _ = kdtree.query(xy)
    return CONSTANT_SPEED / DMAX * np.clip(d, DMIN, DMAX)


def plot_all(model, occupied, kdtree, free_xy, bounds, goal, out="field.png"):
    (xmin, ymin), (xmax, ymax) = bounds

    xs = np.linspace(xmin, xmax, GRID_N)
    ys = np.linspace(ymin, ymax, GRID_N)
    X, Y = np.meshgrid(xs, ys)
    grid = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)

    goal_rep = np.repeat(np.array([goal], dtype=np.float32), len(grid), axis=0)
    qg_fixed = mx.array(goal_rep)

    # only trust the field where free-space data exists
    free_tree = cKDTree(free_xy)
    d_free, _ = free_tree.query(grid)
    mask = (d_free < MASK_RADIUS).reshape(GRID_N, GRID_N)

    T = np.array(model.compute_T(mx.array(grid), qg_fixed)).reshape(GRID_N, GRID_N)

    def T_of_q(x):
        return mx.sum(model.compute_T(x, qg_fixed))

    grad = mx.grad(T_of_q)(mx.array(grid))
    S_pred = np.array(
        1.0 / mx.maximum(1e-8, mx.sqrt(mx.sum(grad * grad, axis=-1)))
    ).reshape(GRID_N, GRID_N)

    S_true = speed_star(grid, kdtree).reshape(GRID_N, GRID_N)

    tau = model.tau(mx.array(grid), qg_fixed)
    tau_masked = np.array(tau).reshape(GRID_N, GRID_N)[mask]

    T = np.where(mask, T, np.nan)
    S_pred = np.where(mask, S_pred, np.nan)
    S_true = np.where(mask, S_true, np.nan)

    extent = [xmin, xmax, ymin, ymax]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))

    im0 = ax[0].imshow(T, origin="lower", extent=extent, cmap="viridis")
    ax[0].contour(X, Y, np.nan_to_num(T, nan=np.nanmax(T)),
                  levels=25, colors="white", linewidths=0.4, alpha=0.6)
    ax[0].set_title("time field  T(q, goal)")
    plt.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(S_pred, origin="lower", extent=extent, cmap="magma",
                       vmin=0, vmax = np.nanmax(S_true))
    ax[1].set_title("predicted speed  S")
    plt.colorbar(im1, ax=ax[1])

    im2 = ax[2].imshow(S_true, origin="lower", extent=extent, cmap="magma",
                       vmin=0, vmax = np.nanmax(S_true))
    ax[2].set_title("ground truth  S*")
    plt.colorbar(im2, ax=ax[2])

    for a in ax:
        a.scatter(occupied[:, 0], occupied[:, 1], s=0.2, c="k", alpha=0.5)
        a.scatter(*goal, c="red", s=80, marker="*", zorder=5)
        a.set_xlim(xmin, xmax)
        a.set_ylim(ymin, ymax)
        a.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print("wrote", out)

    # numbers matter more than the eyeball test -- a near-constant field can
    # still look plausible in colour
    err = np.abs(S_pred - S_true)
    print(f"masked to  {mask.sum()} / {mask.size} grid points")
    print(f"tau        range [{tau_masked.min():.5f}, {tau_masked.max():.5f}]")
    print(f"S error    mean {np.nanmean(err):.4f}   max {np.nanmax(err):.4f}")
    print(f"S_pred     range [{np.nanmin(S_pred):.4f}, {np.nanmax(S_pred):.4f}]")
    print(f"S_true     range [{np.nanmin(S_true):.4f}, {np.nanmax(S_true):.4f}]")
    print(f"T          range [{np.nanmin(T):.4f}, {np.nanmax(T):.4f}]")


if __name__ == "__main__":
    occupied = load_occupied(OCTREE_PATH, RESOLUTION)
    kdtree = cKDTree(occupied)
    print(f"loaded {len(occupied)} occupied points")

    free = np.loadtxt(FREE_PATH)
    free_xy = free[:, :2]
    bounds = (free_xy.min(axis=0), free_xy.max(axis=0))
    print(f"map bounds  x [{bounds[0][0]:.2f}, {bounds[1][0]:.2f}]"
          f"  y [{bounds[0][1]:.2f}, {bounds[1][1]:.2f}]")

    model = NTField()
    model.f_encoder.load_weights("f_encoder.npz")
    model.tao_model.load_weights("tao.npz")
    mx.eval(model.parameters())
    print("loaded weights")

    goal = free_xy[len(free_xy) // 2]
    print(f"goal at {goal}")

    plot_all(model, occupied, kdtree, free_xy, bounds, goal)