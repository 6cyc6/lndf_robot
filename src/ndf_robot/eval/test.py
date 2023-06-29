import numpy as np
import polyscope as ps
from ndf_robot.eval.query_points import QueryPoints


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
#
# query_pts_type = 'RECT'
# query_pts_args = {
#     'n_pts': 1000,
#     'x': 0.08,
#     'y': 0.04,
#     'z1': 0.05,
#     'z2': 0.02,
# }

query_pts_type = 'CYLINDER'
query_pts_args = {
    'n_pts': 1000,
    'radius': 0.02,
    'height': 0.08
}

query = QueryPoints
shelf_pts = query.generate_ndf_shelf(500)
gripper_pts = query.generate_ndf_gripper(500)
print(shelf_pts)

ps.init()
ps.set_up_dir("z_up")
qt = ps.register_point_cloud("shelf", shelf_pts, radius=0.006, enabled=True)
ps.register_point_cloud("gripper", gripper_pts, radius=0.006, enabled=True)
ps.show()
