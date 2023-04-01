import os
import sys
import random
import time

import os.path as osp

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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
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
    # load two point clouds
    dir_pcd_1 = BASE_DIR + '/data/scene_0.npz'
    dir_pcd_2 = BASE_DIR + '/data/scene_3.npz'
    pcd_1 = np.load(dir_pcd_1, allow_pickle=True)
    pcd_2 = np.load(dir_pcd_2, allow_pickle=True)
    pcd1 = pcd_1["pcd"]
    pcd2 = pcd_2["pcd"]

    query_pts = create_query_pts(query_pts_type, query_pts_args)

    grasp_ref = np.array([[-0.75646931, 0.49684362, 0.42532411, -0.04318861],
                          [0.65286087, 0.53478415, 0.53645, 0.8694924],
                          [0.03907517, 0.68348543, -0.72891755, 0.94439177],
                          [0., 0., 0., 1.]])

    query_x = np.random.uniform(-0.02, 0.02, 1000)
    query_y = np.random.uniform(-0.04, 0.04, 1000)
    query_z = np.random.uniform(-0.05 + 0.1, 0.02 + 0.1, 1000)
    ones = np.ones(1000)
    ref_pts_gripper = np.vstack([query_x, query_y, query_z])
    ref_pts_gripper = ref_pts_gripper.T
    hom_query_pts = np.vstack([query_x, query_y, query_z, ones])

    # transform
    ref_query_pts = grasp_ref @ hom_query_pts
    ref_query_pts = ref_query_pts[:3, :]
    ref_query_pts = ref_query_pts.T

    # ps.init()
    # ps.set_up_dir("z_up")
    # ps1 = ps.register_point_cloud("pcd_1", pcd1, radius=0.006, enabled=True)
    # ps2 = ps.register_point_cloud("pcd_2", ref_query_pts, radius=0.006, enabled=True)
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
        'opt_iterations': 1000,
        'rand_translate': True,
        'use_tsne': False,
        'M_override': 20,
    }

    opt_viz_path = 'temp'

    optimizer = OccNetOptimizer(model, query_pts, viz_path=opt_viz_path,
                                **optimizer_args)
    pcd1_ = optimizer.shape_completion(pcd1, thresh=0.3)
    pcd2_ = optimizer.shape_completion(pcd2, thresh=0.3)

    optimizer.compute_target_act(pcd1_, ref_query_pts)

    # ps.init()
    # ps.set_up_dir("z_up")
    # ps1 = ps.register_point_cloud("pcd_1", pcd1, radius=0.006, enabled=True)
    # ps2 = ps.register_point_cloud("pcd_2", pcd1_, radius=0.006, enabled=True)
    # ps.show()

    # optimize
    pose_mats, best_idx, intermediates = optimizer.optimize_transform_implicit(
        pcd2, ee=True, viz_path=opt_viz_path, return_intermediates=True)

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
    ps1 = ps.register_point_cloud("pcd_1", pcd2_, radius=0.006, enabled=True)
    ps2 = ps.register_point_cloud("pcd_2", final_query_pts, radius=0.006, enabled=True)
    ps.register_point_cloud("pcd_3", pcd1_, radius=0.006, enabled=True)
    ps.register_point_cloud("pcd_4", ref_query_pts, radius=0.006, enabled=True)
    ps.show()
