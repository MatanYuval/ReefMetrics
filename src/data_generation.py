import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def xyz2pcd(xyz, colors=None, vis=True, save=None):
    if colors is None:
        colors = get_z_colors(xyz)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if vis:
            o3d.visualization.draw_geometries([pcd])
        if save is not None:
            o3d.io.write_point_cloud(f"data\\{save}.pcd", pcd)

        return pcd


def get_z_colors(xyz, cmap="viridis"):
    return plt.get_cmap(cmap)(xyz[:, 2])[:, :-1]


def gen_cube_xyz(n):
    xyz = np.random.rand(n, 3)
    return xyz


def gen_box_xyz(n):
    a = 1
    xy = np.stack((np.random.rand(n), np.random.rand(n), np.random.choice([0, a], size=n)), axis=1)
    yz = np.stack((np.random.choice([0, a], size=n), np.random.rand(n), np.random.rand(n)), axis=1)
    xz = np.stack((np.random.rand(n), np.random.choice([0, a], size=n), np.random.rand(n)), axis=1)

    xyz = np.concatenate((xy, yz, xz))
    return xyz


def gen_shape(shape, n, vis=True, save=None):
    if shape == "cube":
        xyz_fn = gen_cube_xyz
    elif shape == "box":
        xyz_fn = gen_box_xyz
    else:
        raise NotImplementedError

    xyz = xyz_fn(n)

    if save is not None:
        save = shape
    pcd = xyz2pcd(xyz, vis=vis, save=save)
    return pcd



