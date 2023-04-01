import os
import random
import time

import os.path as osp

from IPython.display import clear_output

import polyscope as ps
import numpy as np
import torch
from torch.nn import functional as F
import trimesh
from trimesh import viewer

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go

from scipy.spatial.transform import Rotation as R
import plotly.express as px

from ndf_robot.utils import path_util, util, torch_util
from ndf_robot.utils.plotly_save import multiplot

import ndf_robot.model.vnn_occupancy_net.vnn_occupancy_net_pointnet_dgcnn \
    as vnn_occupancy_network
import ndf_robot.model.conv_occupancy_net.conv_occupancy_net \
    as conv_occupancy_network

from ndf_robot.opt.optimizer_lite import OccNetOptimizer

from ndf_robot.eval.query_points import QueryPoints

from ndf_robot.eval.demo_io import DemoIO


# Choose query points from these types.
QueryPointTypes = {
    'SPHERE',
    'RECT',
    'CYLINDER',
    'ARM',
    'SHELF',
    'NDF_GRIPPER',
    'NDF_RACK',
    'NDF_SHELF',
}

query_pts_type = 'RECT'
query_pts_args = {
    'n_pts': 1000,
    'x': 0.08,
    'y': 0.04,
    'z1': 0.05,
    'z2': 0.02,
}


def create_query_pts(query_pts_type, query_pts_args) -> np.ndarray:
    """
    Create query points from given config

    Args:
        query_pts_config(dict): Configs loaded from yaml file.

    Returns:
        np.ndarray: Query point as ndarray
    """

    assert query_pts_type in QueryPointTypes, 'Invalid query point type'

    if query_pts_type == 'SPHERE':
        query_pts = QueryPoints.generate_sphere(**query_pts_args)
    elif query_pts_type == 'RECT':
        query_pts = QueryPoints.generate_rect(**query_pts_args)
    elif query_pts_type == 'CYLINDER':
        query_pts = QueryPoints.generate_cylinder(**query_pts_args)
    elif query_pts_type == 'ARM':
        query_pts = QueryPoints.generate_rack_arm(**query_pts_args)
    elif query_pts_type == 'SHELF':
        query_pts = QueryPoints.generate_shelf(**query_pts_args)
    elif query_pts_type == 'NDF_GRIPPER':
        query_pts = QueryPoints.generate_ndf_gripper(**query_pts_args)
    elif query_pts_type == 'NDF_RACK':
        query_pts = QueryPoints.generate_ndf_rack(**query_pts_args)
    elif query_pts_type == 'NDF_SHELF':
        query_pts = QueryPoints.generate_ndf_shelf(**query_pts_args)

    return query_pts


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    random.seed(seed)

    use_random_rotation = True

    # -- Some objects to try -- #
    obj_model = osp.join(path_util.get_ndf_obj_descriptions(), 'mug_std_centered_obj_normalized/e94e46bc5833f2f5e57b873e4f3ef3a4/models/model_normalized.obj')
    # obj_model = osp.join(path_util.get_ndf_obj_descriptions(), 'bowl_handle_std_centered_obj_normalized/34875f8448f98813a2c59a4d90e63212-h/models/model_normalized.obj')
    # obj_model = osp.join(path_util.get_ndf_obj_descriptions(), 'bottle_handle_std_centered_obj_normalized/f853ac62bc288e48e56a63d21fb60ae9-h/models/model_normalized.obj')

    obj_scale = 1.0
    n_samples = 1500
    # assert osp.exists(obj_model), 'Object model not found'

    obj_mesh = trimesh.load(obj_model, process=False)
    obj_mesh.apply_scale(obj_scale)
    obj_pcd = obj_mesh.sample(n_samples)

    upright_rotation = np.eye(4)
    upright_rotation[:3, :3] = util.make_rotation_matrix('x', np.pi/2)
    obj_pcd = util.transform_pcd(obj_pcd, upright_rotation)

    random_rotation = np.eye(4)
    if use_random_rotation:
        random_rotation[:3, :3] = R.random().as_matrix()

    rotated_obj_pcd = util.transform_pcd(obj_pcd, random_rotation)

    query_pts = create_query_pts(query_pts_type, query_pts_args)

    # ps.init()
    # ps.set_up_dir("z_up")
    # ps1 = ps.register_point_cloud("pcd_1", obj_pcd, radius=0.006, enabled=True)
    # ps2 = ps.register_point_cloud("pcd_2", rotated_obj_pcd, radius=0.006, enabled=True)
    # qt = ps.register_point_cloud("pcd_3", query_pts, radius=0.006, enabled=True)
    # ps.show()

    # model
    model_args = {
        'latent_dim': 128,  # Number of voxels in convolutional occupancy network
        'model_type': 'pointnet',  # Encoder type
        'return_features': True,  # Return latent features for evaluation
        'sigmoid': False,  # Use sigmoid activation on last layer
        'acts': 'last',  # Return last activations of occupancy network
    }
    model_checkpoint = 'lndf_weights.pth'
    model_checkpoint_path = osp.join(path_util.get_ndf_model_weights(), model_checkpoint)

    model = conv_occupancy_network.ConvolutionalOccupancyNetwork(**model_args)
    model.load_state_dict(torch.load(model_checkpoint_path))

    # optimizer
    optimizer_args = {
        'opt_iterations': 500,
        'rand_translate': True,
        'use_tsne': False,
        'M_override': 20,
    }

    opt_viz_path = 'temp'

    optimizer = OccNetOptimizer(model, query_pts, viz_path=opt_viz_path,
                                **optimizer_args)

    demo_exp = 'lndf_mug_handle_demos'
    n_demos = 10

    demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', demo_exp)
    demo_fnames = os.listdir(demo_load_dir)

    assert len(demo_fnames), 'No demonstrations found in path: %s!' \
                             % demo_load_dir

    # Sort out grasp demos
    grasp_demo_fnames = [osp.join(demo_load_dir, fn) for fn in
                         demo_fnames if 'grasp_demo' in fn]

    demo_shapenet_ids = []
    demo_list = []

    # Iterate through all demos, extract relevant information and
    # prepare to pass into optimizer
    random.shuffle(grasp_demo_fnames)
    for grasp_demo_fn in grasp_demo_fnames[:n_demos]:
        print('Loading grasp demo from fname: %s' % grasp_demo_fn)
        grasp_data = np.load(grasp_demo_fn, allow_pickle=True)

        demo = DemoIO.process_grasp_data(grasp_data)
        demo_list.append(demo)

        optimizer.add_demo(demo)
        demo_shapenet_ids.append(demo.obj_shapenet_id)

    optimizer.process_demos()

    # visualization
    sample_demo = demo_list[0]
    demo_obj_pts = sample_demo.obj_pts
    # demo_query_pts = sample_demo.query_pts
    demo_query_pts = query_pts
    demo_obj_pose = sample_demo.obj_pose_world
    demo_query_pose = sample_demo.query_pose_world

    posed_obj_pts = util.apply_pose_numpy(demo_obj_pts, demo_obj_pose)
    posed_query_pts = util.apply_pose_numpy(demo_query_pts, demo_query_pose)

    # ps.init()
    # ps.set_up_dir("z_up")
    # ps1 = ps.register_point_cloud("pcd_1", posed_obj_pts, radius=0.006, enabled=True)
    # ps2 = ps.register_point_cloud("pcd_2", posed_query_pts, radius=0.006, enabled=True)
    # ps.show()

    # optimize
    pose_mats, best_idx, intermediates = optimizer.optimize_transform_implicit(
        rotated_obj_pcd, ee=True, viz_path=opt_viz_path, return_intermediates=True)

    # visualization
    idx = best_idx
    best_pose_mat = pose_mats[idx]

    final_query_pts = util.transform_pcd(query_pts, best_pose_mat)

    # Generate trail of intermediate optimization results
    intermediate_query_pts = []
    query_pts_mean = np.mean(query_pts, axis=0).reshape(1, 3)
    for iter_mats in intermediates:
        iter_query_pts = util.transform_pcd(query_pts_mean, iter_mats[idx])
        intermediate_query_pts.append(iter_query_pts)

    # Plot
    ps.init()
    ps.set_up_dir("z_up")
    ps1 = ps.register_point_cloud("pcd_1", rotated_obj_pcd, radius=0.006, enabled=True)
    ps2 = ps.register_point_cloud("pcd_2", final_query_pts, radius=0.006, enabled=True)
    ps.show()
