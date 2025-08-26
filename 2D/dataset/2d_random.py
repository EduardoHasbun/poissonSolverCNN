import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as rgi
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def load_config(cfg_path: str) -> dict:
    """Load YAML configuration file."""
    with open(cfg_path, "r") as yaml_stream:
        return yaml.safe_load(yaml_stream)


def generate_random(cfg: dict, seed: int = None) -> np.ndarray:
    """
    Generate a random dataset sample using interpolation from a lower-resolution grid.
    """
    if seed is not None:
        np.random.seed(seed)

    # Extract domain parameters
    xmin, xmax, nnx = cfg["domain"]["xmin"], cfg["domain"]["xmax"], cfg["domain"]["nnx"]
    ymin, ymax, nny = cfg["domain"]["ymin"], cfg["domain"]["ymax"], cfg["domain"]["nny"]
    n_res_factor = cfg.get("n_res_factor", 16)

    # Coarse grid
    nnx_low, nny_low = nnx // n_res_factor, nny // n_res_factor
    x_low = np.linspace(xmin, xmax, nnx_low)
    y_low = np.linspace(ymin, ymax, nny_low)

    # Full-resolution grid
    x = np.linspace(xmin, xmax, nnx)
    y = np.linspace(ymin, ymax, nny)
    points = np.array(np.meshgrid(x, y, indexing="ij")).T.reshape(-1, 2)

    # Random field on coarse grid
    z_low = 2 * np.random.random((nnx_low, nny_low)) - 1

    # Interpolate to fine grid
    interpolator = rgi((x_low, y_low), z_low, method="cubic")
    return interpolator(points).reshape((nnx, nny))


def plot_sample(data: np.ndarray, save_path: str) -> None:
    """Plot a single data sample and save it as an image."""
    plt.figure(figsize=(8, 6))
    plt.imshow(data, origin="lower", cmap="bwr")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def main(cfg_path: str, plot: bool = False) -> None:
    cfg = load_config(cfg_path)

    nits = cfg["n_it"]
    name = cfg["name"]
    nnx, nny = cfg["domain"]["nnx"], cfg["domain"]["nny"]

    # Prepare directories
    os.makedirs("generated", exist_ok=True)
    plots_dir = os.path.join("generated", "plots")
    if plot:
        os.makedirs(plots_dir, exist_ok=True)

    print(f"Grid size: {nnx} x {nny}, xmax: {cfg['domain']['xmax']}, ymax: {cfg['domain']['ymax']}")

    # Generate data
    data_array = np.empty((nits, nnx, nny))

    with Pool(processes=cpu_count()) as pool:
        for idx, data in tqdm(enumerate(pool.imap(lambda i: generate_random(cfg), range(nits))),
                              total=nits, desc="Processing"):
            data_array[idx] = data
            if plot:
                plot_sample(data, os.path.join(plots_dir, f"random_data_plot_{idx}.png"))

    # Save dataset
    np.save(os.path.join("generated", name), data_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RHS random dataset generator")
    parser.add_argument("-c", "--cfg", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--plot", action="store_true", help="Enable plotting of samples")
    args = parser.parse_args()

    main(args.cfg, args.plot)
