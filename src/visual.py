import pickle
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

from src.io_utils import get_models_list
from src.local_setting import RESULT_DIR
import seaborn as sns

from src.setting import PCD_MODELS


def create_box(size, factor, trnaslate):
    size_in_practice = factor * size
    residual = (1 - factor) * size
    mesh_box0 = o3d.geometry.TriangleMesh.create_box(width=size_in_practice,
                                                     height=size_in_practice,
                                                     depth=size_in_practice)
    mesh_box0.translate(np.array([residual / 2] * 3) + trnaslate)
    mesh_box0.compute_vertex_normals()
    mesh_box0.paint_uniform_color([0.1, 0.8, 0.8])

    return mesh_box0


def general_vis_counting_box(coor, pcd, pc_max_size, size, margin_factor):
    row_cubes = int(np.ceil(pc_max_size / size))
    cubes = []
    for i in range(row_cubes):
        for j in range(row_cubes):
            for k in range(row_cubes):
                cube_axis = np.array([i, j, k]) * size
                cube_limits = cube_axis + size
                relevant_points = np.logical_and(coor > cube_axis, coor < cube_limits)
                if relevant_points.prod(1).sum() > 0:
                    cubes.append(create_box(size, margin_factor, cube_axis))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for cube in cubes:
        vis.add_geometry(cube)
    # vis.get_render_option().load_from_json(
    #     r"D:\Naama\Documents\Projects\deeplab-v3-transfer-learning\renderoption1.json")
    vis.run()
    vis.destroy_window()


def add_3d_axes(fig, rows, cols, pos):
    ax = fig.add_subplot(rows, cols, pos, projection='3d')
    # ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return ax


def plot_trial(fig, trials, t, coor, pc_max_size):
    ax = add_3d_axes(fig, 1, trials, t + 1)
    ax.scatter3D(coor[:, 0], coor[:, 1], coor[:, 2], c=coor[:, 2], zorder=-1)
    ax.plot([0, 0, 0, 0, 0], [0, 0, pc_max_size, pc_max_size, 0], [0, pc_max_size, pc_max_size, 0, 0], c='k', zorder=1)
    ax.plot([0, pc_max_size, pc_max_size, 0, 0], [0, 0, 0, 0, 0], [0, 0, pc_max_size, pc_max_size, 0], c='k', zorder=1)
    ax.plot([pc_max_size, pc_max_size, pc_max_size, pc_max_size, pc_max_size], [0, pc_max_size, pc_max_size, 0, 0],
            [0, 0, pc_max_size, pc_max_size, 0], c='k', zorder=1)
    ax.plot([pc_max_size, 0, 0, pc_max_size, pc_max_size],
            [pc_max_size, pc_max_size, pc_max_size, pc_max_size, pc_max_size], [0, 0, pc_max_size, pc_max_size, 0],
            c='k', zorder=1)


class BoxCountingResults:
    def __init__(self, model, year, pcd_path):
        self.year = year
        self.model = model
        self.pcd_path = pcd_path

        self.counts_list = []
        self.lengths_list = []
        self.box_counting_dicts = []

        self.box_counting_slope = None
        self.coeffs = None

    def append(self, res):
        self.box_counting_dicts.append(res)

        sorted_res = list(zip(*sorted(res.items(), key=lambda e: e[0])))
        self.lengths_list.append(list(sorted_res[0]))
        self.counts_list.append(list(sorted_res[1]))

    def get_log_values(self):
        lengths_array = np.concatenate(self.lengths_list)
        counts_array = np.concatenate(self.counts_list)

        # TODO need to decide which log base to work with
        xs = np.log10(1 / lengths_array)
        ys = np.log10(counts_array)

        return xs, ys

    def calc_box_counting_slope(self):
        xs, ys = self.get_log_values()
        coeffs, _ = np.polynomial.polynomial.polyfit(xs, ys, 1, full=True)
        self.coeffs = coeffs
        self.box_counting_slope =  coeffs[1]
        return coeffs[1]

    def __len__(self):
        return len(self.box_counting_dicts)


def plot_box_counting_results_aggregate(model_res: List[BoxCountingResults], show=True, save=True):
    legend = []
    fig, ax = plt.subplots(figsize=(8, 6))

    for box_counting_res in model_res:
        xs, ys = box_counting_res.get_log_values()
        slope  = box_counting_res.box_counting_slope
        coeffs = box_counting_res.coeffs
        print(f"coefficient (fractal dimension) = {slope}")
        legend += [f"{box_counting_res.year}: Fitted line {coeffs[1]:.3f}X+{coeffs[0]:.3f}",
                   f"{box_counting_res.year} Measured points"]
        sns.regplot(x=xs, y=ys, ax=ax,
                    label=f"{box_counting_res.year}: Fitted line {coeffs[1]:.3f}X+{coeffs[0]:.3f}")

    ax.set_ylabel("$\log N(\epsilon)$")
    ax.set_xlabel("$\log 1/ \epsilon$")
    ax.set_title(f"Box counting fit: {model_res[0].model} (fit over {len(model_res[0])} repetitions)")
    ax.legend()
    plt.show()

    if save:
        plt.savefig(str(RESULT_DIR / f"{model_res[0].model}__fig.png"))
        for res in model_res:
            with open(str(RESULT_DIR / f"{res.pcd_path.stem}__res.txt"), 'w') as f:
                f.write("Box counting dictionaries list:\n")
                f.write(str(res.box_counting_dicts))
                f.write("\n")
                f.write("Calculated slope:\n")
                f.write(str(res.box_counting_slope))
            with open(str(RESULT_DIR / f"{res.pcd_path.stem}__res.pckl"), 'wb') as f:
                pickle.dump(res, f)


if __name__ == '__main__':
    models_name_list = ['Kzaa']
    model_type = 'mesh'

    models_list = get_models_list(models_name_list, model_type)
    for model, paths in models_list.items():
        res_list = []
        for year, pcd_name in paths.items():
            with open(str(RESULT_DIR/f'{Path(pcd_name).stem}__res.pckl'), 'rb') as f:
                res_list.append(pickle.load(f))

        plot_box_counting_results_aggregate(res_list)





