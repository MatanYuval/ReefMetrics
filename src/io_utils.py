import open3d as o3d

from src.local_setting import DATA_ROOT
from src.setting import PCD_MODELS, MESH_MODELS


def read_model(pcd_name, model_type):
    pcd_path = DATA_ROOT / pcd_name
    assert pcd_path.exists()
    if model_type == 'pcd':
        pcd = o3d.io.read_point_cloud(str(pcd_path))
    elif model_type == 'mesh':
        mesh = o3d.io.read_triangle_mesh(str(pcd_path))
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        # TODO maybe sample the mesh byAAA
        #  pcd = mesh.sample_points_uniformly(number_of_points=50000)
    else:
        raise NotImplementedError(f"model_type: {model_type}")

    return pcd


def get_models_list(models_names_list, model_type):
    if model_type == 'pcd':
        models = {name: PCD_MODELS[name] for name in models_names_list}
    elif model_type == 'mesh':
        models = {name: MESH_MODELS[name] for name in models_names_list}
    else:
        raise NotImplementedError(f"model_type: {model_type}")

    return models