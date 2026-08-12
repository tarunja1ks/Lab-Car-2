"""
Export trained MLX weights to a flat text format that C++ can read without
any dependencies.

npz is a zip archive of .npy files -- readable in C++ only via a third-party
library (cnpy, xtensor-io). Since the weights are written once and read many
times, a plain text dump is simpler and has no build-system cost.

Format:
    <name> <rows> <cols>
    v v v v ...          (rows*cols values, row-major, whitespace separated)

Usage:
    python export_weights.py f_encoder.npz tao.npz ntfield_weights.txt
"""

# python export_weights.py f_encoder.npz tao.npz ntfield_weights.txt

import sys
import numpy as np


def load_npz_ordered(path):
    """
    Load an MLX-saved npz and return (key, array) pairs in layer order.

    MLX names Sequential parameters "layers.N.weight" / "layers.N.bias", where
    N indexes into the Sequential including activation layers. Sorting by that
    integer keeps the layers in forward order regardless of how many
    activations sit between them.
    """
    data = np.load(path)

    def layer_index(key):
        parts = key.split(".")
        for p in parts:
            if p.isdigit():
                return int(p)
        return -1

    keys = sorted(data.files, key=lambda k: (layer_index(k), k))
    return [(k, np.array(data[k])) for k in keys]


def write_matrix(f, name, arr):
    arr = np.atleast_2d(arr)
    if arr.shape[0] == 1 and "bias" in name:
        arr = arr.reshape(-1, 1)          # biases as column vectors

    rows, cols = arr.shape
    f.write(f"{name} {rows} {cols}\n")
    f.write(" ".join(f"{v:.9g}" for v in arr.ravel()))
    f.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    f_path, tao_path, out_path = sys.argv[1:4]

    f_params = load_npz_ordered(f_path)
    tao_params = load_npz_ordered(tao_path)

    with open(out_path, "w") as f:
        f.write(f"{len(f_params) + len(tao_params)}\n")
        for name, arr in f_params:
            write_matrix(f, "f." + name, arr)
        for name, arr in tao_params:
            write_matrix(f, "tao." + name, arr)

    print(f"wrote {out_path}")
    for name, arr in f_params:
        print(f"  f.{name:24s} {arr.shape}")
    for name, arr in tao_params:
        print(f"  tao.{name:22s} {arr.shape}")