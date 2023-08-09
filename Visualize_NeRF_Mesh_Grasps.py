# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Train NeRF Grasp Metric
#
# ## Summary (April 18, 2023)
#
# The purpose of this script is to train a neural network model to take in:
#
# * a NeRF object model
#
# * $n$ ray origins and directions representating fingers approaching (for now, $n = 2$)
#
# and output:
#
# * a grasp metric $g$ representing the quality of grasp (for now, $g \in [0, 1]$, where 0 is failed grasp and 1 is successful grasp).
#
# To do this, we will be using the [ACRONYM dataset](https://sites.google.com/nvidia.com/graspdataset), which contains ~1.7M grasps on over 8k objects each labeled with the grasp success.

# %%
import wandb
import os
import h5py
import numpy as np
from localscope import localscope
import time
import math

import random
import torch
import plotly.graph_objects as go
import trimesh

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d

from nerf_grasping.grasp_utils import nerf_to_ig, ig_to_nerf


# %% [markdown]
# # Read In Config


# %%
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


# %%
# Use with hydra later
# if is_notebook():
#     arguments = []
# else:
#     arguments = sys.argv[1:]
#     print(f"arguments = {arguments}")
#
# OmegaConf.register_new_resolver("eval", eval)
# with initialize(version_base=None, config_path="train_bc_config_files"):
#     cfg = compose(config_name="config", overrides=arguments)
#     print(OmegaConf.to_yaml(cfg))

# %%

# %% [markdown]
# # Setup Wandb

# %%

# time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# run_name = f"{cfg.wandb_name}_{time_str}" if len(cfg.wandb_name) > 0 else time_str

# wandb.init(entity=cfg.wandb_entity,
#            project=cfg.wandb_project,
#            name=run_name,
#            group=cfg.wandb_group if len(cfg.wandb_group) > 0 else None,
#            job_type=cfg.wandb_job_type if len(cfg.wandb_job_type) > 0 else None,
#            config=OmegaConf.to_container(cfg),
#            reinit=True)

# %%


@localscope.mfc
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


# set_seed(cfg.RANDOM_SEED)

# %% [markdown]
# # Load Data From Files

# %%
# TODO: Need way to connect an acronym file to a nerf model nicely
assets_dir_filepath = "/juno/u/tylerlum/github_repos/nerf_grasping/assets/objects"
acronym_dir_filepath = "/juno/u/tylerlum/github_repos/acronym/data/grasps"
USE_MUG = False
if USE_MUG:
    nerf_model_workspace = "isaac_Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682458/"
    acronym_data_filepath = os.path.join(
        acronym_dir_filepath,
        "Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5",
    )
    urdf_filepath = os.path.join(
        assets_dir_filepath, "urdf", "Mug_10f6e09036350e92b3f21f1137c3c347.urdf"
    )
    obj_filepath = os.path.join(
        assets_dir_filepath, "meshes", "Mug", "10f6e09036350e92b3f21f1137c3c347.obj"
    )
else:
    nerf_model_workspace = "isaac_Dog_35f73ca2716aefcfbeccafa1b3b5f850_0.0041745997"
    acronym_data_filepath = os.path.join(
        acronym_dir_filepath,
        "Dog_35f73ca2716aefcfbeccafa1b3b5f850_0.004174599731357345.h5",
    )
    urdf_filepath = os.path.join(
        assets_dir_filepath, "urdf", "Dog_35f73ca2716aefcfbeccafa1b3b5f850.urdf"
    )
    obj_filepath = os.path.join(
        assets_dir_filepath, "meshes", "Dog", "35f73ca2716aefcfbeccafa1b3b5f850.obj"
    )

# %%
acronym_data = h5py.File(acronym_data_filepath, "r")
mesh_scale = float(acronym_data["object/scale"][()])

grasp_transforms = np.array(acronym_data["grasps/transforms"])
grasp_successes = np.array(acronym_data["grasps/qualities/flex/object_in_gripper"])

# %%
print(f"{grasp_transforms.shape = }")
print(f"{grasp_successes.shape = }")

# %%
LEFT_TIP_POSITION_GRASP_FRAME = np.array(
    [4.10000000e-02, -7.27595772e-12, 1.12169998e-01]
)
RIGHT_TIP_POSITION_GRASP_FRAME = np.array(
    [-4.10000000e-02, -7.27595772e-12, 1.12169998e-01]
)


# %%
@localscope.mfc
def position_to_transformed_positions(position, transforms):
    assert position.shape == (3,)
    assert len(transforms.shape) == 3 and transforms.shape[1:] == (4, 4)
    num_transforms = transforms.shape[0]

    transformed_positions = (transforms @ np.array([*position, 1.0]).reshape(1, 4, 1))[
        :, :3, :
    ].squeeze()
    assert transformed_positions.shape == (num_transforms, 3)
    return transformed_positions


@localscope.mfc
def position_to_transformed_positions_unvectorized(position, transforms):
    assert position.shape == (3,)
    assert len(transforms.shape) == 3 and transforms.shape[1:] == (4, 4)
    num_transforms = transforms.shape[0]

    transformed_positions = []
    for i in range(num_transforms):
        transformed_positions.append((transforms[i] @ np.array([*position, 1.0]))[:3])
    transformed_positions = np.stack(transformed_positions)
    return transformed_positions


@localscope.mfc
def run_sanity_check(position, transforms):
    # Non-vectorized
    start = time.time()
    positions_object_frame = position_to_transformed_positions_unvectorized(
        position=position, transforms=transforms
    )
    print(f"Non-vectorized took {1000 * (time.time() - start):.2f} ms")

    # Vectorized version
    start = time.time()
    positions_object_frame_2 = position_to_transformed_positions(
        position=position, transforms=transforms
    )
    print(f"Vectorized took {1000 * (time.time() - start):.2f} ms")

    assert np.max(np.abs(positions_object_frame - positions_object_frame_2)) < 1e-5
    print("Passed the test, they match!")
    return


# %%
run_sanity_check(position=LEFT_TIP_POSITION_GRASP_FRAME, transforms=grasp_transforms)


# %%
@localscope.mfc
def plot_obj(obj_filepath, scale=1.0, offset=None, color="lightpink"):
    if offset is None:
        offset = np.zeros(3)

    # Read in the OBJ file
    with open(obj_filepath, "r") as f:
        lines = f.readlines()

    # Extract the vertex coordinates and faces from the OBJ file
    vertices = []
    faces = []
    for line in lines:
        if line.startswith("v "):
            vertex = [float(i) * scale for i in line.split()[1:4]]
            vertices.append(vertex)
        elif line.startswith("f "):
            face = [int(i.split("/")[0]) - 1 for i in line.split()[1:4]]
            faces.append(face)

    # Convert the vertex coordinates and faces to numpy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)

    assert len(vertices.shape) == 2 and vertices.shape[1] == 3
    assert len(faces.shape) == 2 and faces.shape[1] == 3

    vertices += offset.reshape(1, 3)

    # Create the mesh3d trace
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=0.5,
        name=f"Mesh: {os.path.basename(obj_filepath)}",
    )

    # Create the layout
    coordinates = "Object Coordinates" if np.all(offset == 0) else "Isaac Coordinates"
    layout = go.Layout(
        scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z")),
        showlegend=True,
        title=f"Mesh: {os.path.basename(obj_filepath)} ({coordinates})",
    )

    # Create the figure
    fig = go.Figure(data=[mesh], layout=layout)

    # Return the figure
    return fig


# %%
mesh = trimesh.load(obj_filepath, force="mesh")


@localscope.mfc
def get_mesh_centroid(mesh, scale=1):
    return np.array(mesh.centroid) * scale


# Get bounds of mesh
@localscope.mfc
def get_mesh_bounds(mesh, scale=1):
    min_points, max_points = mesh.bounds
    return np.array(min_points) * scale, np.array(max_points) * scale


min_points_obj_frame, max_points_obj_frame = get_mesh_bounds(mesh, scale=mesh_scale)
print(min_points_obj_frame, max_points_obj_frame)


# %%
# Use this offset for all plots so that the plot is in isaac coordinates
bound_min_z_obj_frame = min_points_obj_frame[2]
mesh_centroid_obj_frame = get_mesh_centroid(mesh, scale=mesh_scale)

USE_ISAAC_COORDINATES = True
if USE_ISAAC_COORDINATES:
    obj_offset = np.array(
        [
            -mesh_centroid_obj_frame[0],
            -mesh_centroid_obj_frame[1],
            -bound_min_z_obj_frame,
        ]
    )
else:
    obj_offset = np.zeros(3)
print(f"USE_ISAAC_COORDINATES: {USE_ISAAC_COORDINATES}")
print(f"Offset: {obj_offset}")

fig = plot_obj(obj_filepath, scale=mesh_scale, offset=obj_offset)
fig.show()


# %%


@localscope.mfc
def get_mesh_centroid_scatter(mesh_centroid, offset=None):
    if offset is None:
        offset = np.zeros(3)

    translated_mesh_centroid = mesh_centroid + offset
    scatter = go.Scatter3d(
        x=[translated_mesh_centroid[0]],
        y=[translated_mesh_centroid[1]],
        z=[translated_mesh_centroid[2]],
        mode="markers",
        marker=dict(size=10, color="black"),
        name="Mesh Centroid",
    )
    return scatter


@localscope.mfc
def get_mesh_origin_lines(offset=None):
    if offset is None:
        offset = np.zeros(3)

    x_lines = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    y_lines = np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]])
    z_lines = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])
    x_lines += offset
    y_lines += offset
    z_lines += offset
    lines = [
        go.Scatter3d(
            x=x_lines[:, 0],
            y=x_lines[:, 1],
            z=x_lines[:, 2],
            mode="lines",
            line=dict(width=2, color="red"),
            name="Mesh Origin X Axis",
        ),
        go.Scatter3d(
            x=y_lines[:, 0],
            y=y_lines[:, 1],
            z=y_lines[:, 2],
            mode="lines",
            line=dict(width=2, color="green"),
            name="Mesh Origin Y Axis",
        ),
        go.Scatter3d(
            x=z_lines[:, 0],
            y=z_lines[:, 1],
            z=z_lines[:, 2],
            mode="lines",
            line=dict(width=2, color="blue"),
            name="Mesh Origin Z Axis",
        ),
    ]
    return lines


# %%
fig = plot_obj(obj_filepath, scale=mesh_scale, offset=obj_offset)
mesh_centroid_scatter = get_mesh_centroid_scatter(
    mesh_centroid_obj_frame, offset=obj_offset
)
mesh_origin_lines = get_mesh_origin_lines(offset=obj_offset)
fig.add_trace(mesh_centroid_scatter)
for mesh_origin_line in mesh_origin_lines:
    fig.add_trace(mesh_origin_line)
fig.show()


# %%
@localscope.mfc
def get_grasp_gripper_lines(grasp_transforms, grasp_successes, offset=None):
    if offset is None:
        offset = np.zeros(3)

    raw_left_tip = np.array([4.10000000e-02, -7.27595772e-12, 1.12169998e-01])
    raw_right_tip = np.array([-4.10000000e-02, -7.27595772e-12, 1.12169998e-01])
    raw_left_knuckle = np.array([4.10000000e-02, -7.27595772e-12, 6.59999996e-02])
    raw_right_knuckle = np.array([-4.10000000e-02, -7.27595772e-12, 6.59999996e-02])
    raw_hand_origin = np.array([0.0, 0.0, 0.0])
    left_tips = position_to_transformed_positions(
        position=raw_left_tip, transforms=grasp_transforms
    )
    right_tips = position_to_transformed_positions(
        position=raw_right_tip, transforms=grasp_transforms
    )
    left_knuckles = position_to_transformed_positions(
        position=raw_left_knuckle, transforms=grasp_transforms
    )
    right_knuckles = position_to_transformed_positions(
        position=raw_right_knuckle, transforms=grasp_transforms
    )
    hand_origins = position_to_transformed_positions(
        position=raw_hand_origin, transforms=grasp_transforms
    )

    assert (
        left_tips.shape
        == right_tips.shape
        == left_knuckles.shape
        == right_knuckles.shape
        == hand_origins.shape
        == (len(grasp_successes), 3)
    )
    left_tips += offset.reshape(1, 3)
    right_tips += offset.reshape(1, 3)
    left_knuckles += offset.reshape(1, 3)
    right_knuckles += offset.reshape(1, 3)
    hand_origins += offset.reshape(1, 3)

    grasp_lines = []
    for i, (
        left_tip,
        right_tip,
        left_knuckle,
        right_knuckle,
        hand_origin,
        grasp_success,
    ) in enumerate(
        zip(
            left_tips,
            right_tips,
            left_knuckles,
            right_knuckles,
            hand_origins,
            grasp_successes,
        )
    ):
        assert grasp_success in [0, 1]
        color = "green" if grasp_success == 1 else "red"

        # left tip => left knuckle => right knuckle => right tip => right_knuckle => btwn knuckles => hand_origin
        btwn_knuckles = (left_knuckle + right_knuckle) / 2

        points = np.stack(
            [
                left_tip,
                left_knuckle,
                right_knuckle,
                right_tip,
                btwn_knuckles,
                hand_origin,
            ],
            axis=0,
        )
        assert points.shape == (6, 3)

        # Create 1 continous line per grasp
        grasp_line = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="lines",
            line=dict(color=color, width=5),
            name=f"Grasp {i}",
        )
        grasp_lines.append(grasp_line)
    return grasp_lines


# %%
fig = plot_obj(obj_filepath, scale=mesh_scale, offset=obj_offset)
grasp_lines = get_grasp_gripper_lines(
    grasp_transforms[:6], grasp_successes[:6], offset=obj_offset
)
for grasp_line in grasp_lines:
    fig.add_trace(grasp_line)
fig.show()


# %%
@localscope.mfc
def get_grasp_ray_lines(grasp_transforms, grasp_successes, offset=None):
    if offset is None:
        offset = np.zeros(3)

    raw_left_tip = np.array([4.10000000e-02, -7.27595772e-12, 1.12169998e-01])
    raw_right_tip = np.array([-4.10000000e-02, -7.27595772e-12, 1.12169998e-01])
    left_tips = position_to_transformed_positions(
        position=raw_left_tip, transforms=grasp_transforms
    )
    right_tips = position_to_transformed_positions(
        position=raw_right_tip, transforms=grasp_transforms
    )

    assert left_tips.shape == right_tips.shape == (len(grasp_successes), 3)
    left_tips += offset.reshape(1, 3)
    right_tips += offset.reshape(1, 3)

    grasp_lines = []
    for i, (
        left_tip,
        right_tip,
        grasp_success,
    ) in enumerate(
        zip(
            left_tips,
            right_tips,
            grasp_successes,
        )
    ):
        assert grasp_success in [0, 1]
        color = "green" if grasp_success == 1 else "red"

        # left tip => right_tip
        points = np.stack(
            [
                left_tip,
                right_tip,
            ],
            axis=0,
        )
        assert points.shape == (2, 3)

        # Create 1 continous line per grasp
        grasp_line = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="lines",
            line=dict(color=color, width=5),
            name=f"Grasp {i}",
        )
        grasp_lines.append(grasp_line)
    return grasp_lines


# %%
fig = plot_obj(obj_filepath, scale=mesh_scale, offset=obj_offset)
grasp_lines = get_grasp_ray_lines(
    grasp_transforms[:6], grasp_successes[:6], offset=obj_offset
)
for grasp_line in grasp_lines:
    fig.add_trace(grasp_line)
fig.show()

# %% [markdown]
# ## TEMP: Visualize NeRF


# %%
@localscope.mfc(
    allowed=["LEFT_TIP_POSITION_GRASP_FRAME", "RIGHT_TIP_POSITION_GRASP_FRAME"]
)
def get_grasp_query_points_grasp_frame():
    # Grid of points in grasp frame (x, y, z)
    GRIPPER_WIDTH_MM = 82
    GRIPPER_FINGER_WIDTH_MM = 20
    GRIPPER_FINGER_HEIGHT_MM = 36

    # Want points equally spread out in space
    DIST_BTWN_PTS_MM = 1

    NUM_PTS_X = int(GRIPPER_WIDTH_MM / DIST_BTWN_PTS_MM)
    NUM_PTS_Y = int(GRIPPER_FINGER_WIDTH_MM / DIST_BTWN_PTS_MM)
    NUM_PTS_Z = int(GRIPPER_FINGER_HEIGHT_MM / DIST_BTWN_PTS_MM)

    assert NUM_PTS_X * DIST_BTWN_PTS_MM == GRIPPER_WIDTH_MM
    assert NUM_PTS_Y * DIST_BTWN_PTS_MM == GRIPPER_FINGER_WIDTH_MM
    assert NUM_PTS_Z * DIST_BTWN_PTS_MM == GRIPPER_FINGER_HEIGHT_MM
    num_pts = (NUM_PTS_X + 1) * (NUM_PTS_Y + 1) * (NUM_PTS_Z + 1)
    print(f"num_pts: {num_pts}")

    GRIPPER_WIDTH_M = GRIPPER_WIDTH_MM / 1000.0
    GRIPPER_FINGER_WIDTH_M = GRIPPER_FINGER_WIDTH_MM / 1000.0
    GRIPPER_FINGER_HEIGHT_M = GRIPPER_FINGER_HEIGHT_MM / 1000.0

    # Create grid of points in grasp frame with shape (NUM_PTS_X, NUM_PTS_Y, NUM_PTS_Z, 3)
    # So that grid_of_points[2, 3, 5] = [x, y, z], where x, y, z are the coordinates of the point
    x_coords = np.linspace(-GRIPPER_WIDTH_M / 2, GRIPPER_WIDTH_M / 2, NUM_PTS_X + 1)
    y_coords = np.linspace(
        -GRIPPER_FINGER_WIDTH_M / 2, GRIPPER_FINGER_WIDTH_M / 2, NUM_PTS_Y + 1
    )
    z_coords = np.linspace(0, GRIPPER_FINGER_HEIGHT_M, NUM_PTS_Z + 1)

    # Offset so centered between LEFT_TIP_POSITION_GRASP_FRAME and RIGHT_TIP_POSITION_GRASP_FRAME
    center_point = (LEFT_TIP_POSITION_GRASP_FRAME + RIGHT_TIP_POSITION_GRASP_FRAME) / 2
    x_coords += center_point[0]
    y_coords += center_point[1]
    z_coords += center_point[2]

    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    assert (
        xx.shape
        == yy.shape
        == zz.shape
        == (NUM_PTS_X + 1, NUM_PTS_Y + 1, NUM_PTS_Z + 1)
    )
    grid_of_points = np.stack([xx, yy, zz], axis=-1)
    assert grid_of_points.shape == (NUM_PTS_X + 1, NUM_PTS_Y + 1, NUM_PTS_Z + 1, 3)
    return grid_of_points


grasp_query_points_grasp_frame = get_grasp_query_points_grasp_frame()


# %%
# Transform query points to object frame


@localscope.mfc
def get_transformed_points(points, transform):
    assert len(points.shape) == 2 and points.shape[1] == 3
    assert transform.shape == (4, 4)

    points_homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

    # First (4, 4) @ (4, N) = (4, N)
    # Then transpose to get (N, 4)
    transformed_points = np.matmul(transform, points_homogeneous.T).T

    return transformed_points[:, :3]


transform = grasp_transforms[4]
grasp_query_points_object_frame = get_transformed_points(
    grasp_query_points_grasp_frame.reshape(-1, 3), transform
).reshape(grasp_query_points_grasp_frame.shape)


# %%
# Visualize points
@localscope.mfc
def get_points_scatter(points, offset=None):
    if offset is None:
        offset = np.zeros(3)

    assert len(points.shape) == 2 and points.shape[1] == 3

    points_to_plot = points + offset.reshape(1, 3)

    # Use plotly to make scatter3d plot
    scatter = go.Scatter3d(
        x=points_to_plot[:, 0],
        y=points_to_plot[:, 1],
        z=points_to_plot[:, 2],
        mode="markers",
        marker=dict(size=5, color="blue"),
        name="Query Points",
    )

    return scatter


# %%
scatter = get_points_scatter(
    grasp_query_points_object_frame.reshape(-1, 3), offset=obj_offset
)

# Add the scatter plot to a figure and display it
fig = plot_obj(obj_filepath, scale=mesh_scale, offset=obj_offset)
fig.add_trace(scatter)
fig.show()

# %%
from pathlib import Path
from nerf_grasping import nerf_utils
from nerf import utils


def get_root_dir():
    root_dir = None
    try:
        root_dir = Path(os.path.abspath(__file__)).parents[0]
    except:
        pass
    try:
        root_dir = Path(os.path.abspath("."))
    except:
        pass
    if root_dir is None:
        raise ValueError("Can't get path to this file")
    return root_dir


root_dir = get_root_dir()
print(root_dir)


# %%
@localscope.mfc(allowed=["root_dir"])
def load_nerf(workspace, bound, scale):
    parser = utils.get_config_parser()
    opt = parser.parse_args(
        [
            "--workspace",
            f"{root_dir}/nerf_checkpoints/{workspace}",
            "--fp16",
            "--test",
            "--bound",
            f"{bound}",
            "--scale",
            f"{scale}",
            "--mode",
            "blender",
            f"{root_dir}/torch-ngp",
        ]
    )
    # Use options to determine proper network structure.
    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        assert opt.fp16, "tcnn mode must be used with fp16 mode"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    # Create uninitialized network.
    model = NeRFNetwork(
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
    )

    # Create trainer with NeRF; use its constructor to load network weights from file.
    trainer = utils.Trainer(
        "ngp",
        vars(opt),
        model,
        workspace=opt.workspace,
        criterion=None,
        fp16=opt.fp16,
        metrics=[None],
        use_checkpoint="latest",
    )
    assert len(trainer.stats["checkpoints"]) != 0, "failed to load checkpoint"
    return trainer.model


nerf_model = load_nerf(workspace=nerf_model_workspace, bound=2, scale=1)


# %%
nerf_model


# %%
@localscope.mfc
def get_nerf_densities(nerf_model, query_points):
    """
    Evaluates density of a batch of grasp points, shape [B, n_f, 3].
    query_points is torch.Tensor in nerf frame
    """
    B, n_f, _ = query_points.shape
    query_points = query_points.reshape(1, -1, 3)

    return nerf_model.density(query_points).reshape(B, n_f)


grasp_query_points_isaac_frame = np.copy(grasp_query_points_object_frame).reshape(
    -1, 3
) + obj_offset.reshape(1, 3)
grasp_query_points_nerf_frame = ig_to_nerf(
    grasp_query_points_isaac_frame, return_tensor=True
)

nerf_densities_torch = get_nerf_densities(
    nerf_model=nerf_model,
    query_points=grasp_query_points_nerf_frame.reshape(1, -1, 3).float().cuda(),
).reshape(grasp_query_points_object_frame.shape[:-1])

# %%
nerf_densities = nerf_densities_torch.detach().cpu().numpy()

print(f"np.max(nerf_densities) = {np.max(nerf_densities)}")
print(f"np.min(nerf_densities) = {np.min(nerf_densities)}")


# %%
# Visualize points
@localscope.mfc
def get_colored_points_scatter(points, colors, offset=None):
    if offset is None:
        offset = np.zeros(3)

    assert len(points.shape) == 2 and points.shape[1] == 3

    points_to_plot = points + offset.reshape(1, 3)

    # Use plotly to make scatter3d plot
    scatter = go.Scatter3d(
        x=points_to_plot[:, 0],
        y=points_to_plot[:, 1],
        z=points_to_plot[:, 2],
        mode="markers",
        marker=dict(
            size=5,
            color=colors,
            colorscale="viridis",
            colorbar=dict(title="Density Scale"),
        ),
        name="Query Points Densities",
    )

    return scatter


# %%
colored_points_scatter = get_colored_points_scatter(
    points=grasp_query_points_object_frame.reshape(-1, 3),
    colors=nerf_densities.reshape(-1),
    offset=obj_offset,
)

# Add the scatter plot to a figure and display it
fig = plot_obj(obj_filepath, scale=mesh_scale, offset=obj_offset)
fig.add_trace(colored_points_scatter)

# Avoid legend overlap
fig.update_layout(legend_orientation="h")

fig.show()


# %%
@localscope.mfc
def get_query_points_mesh_region(min_points, max_points, n_pts_per_dim):
    """
    Returns a batch of query points in the mesh region.
    """
    x = np.linspace(min_points[0], max_points[0], n_pts_per_dim)
    y = np.linspace(min_points[1], max_points[1], n_pts_per_dim)
    z = np.linspace(min_points[2], max_points[2], n_pts_per_dim)
    xv, yv, zv = np.meshgrid(x, y, z)
    return np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)


query_points_mesh_region_obj_frame = get_query_points_mesh_region(
    min_points_obj_frame, max_points_obj_frame, n_pts_per_dim=100
)

# %%
query_points_mesh_region_obj_frame.shape

# %%
query_points_mesh_region_isaac_frame = np.copy(
    query_points_mesh_region_obj_frame
).reshape(-1, 3) + obj_offset.reshape(1, 3)
query_points_mesh_region_nerf_frame = ig_to_nerf(
    query_points_mesh_region_isaac_frame.reshape(-1, 3), return_tensor=True
)

# %%
nerf_densities_torch = get_nerf_densities(
    nerf_model, query_points_mesh_region_nerf_frame.reshape(1, -1, 3).float().cuda()
).reshape(query_points_mesh_region_nerf_frame.shape[:-1])
nerf_densities = nerf_densities_torch.detach().cpu().numpy()

# %%
points = query_points_mesh_region_obj_frame.reshape(-1, 3)
densities = nerf_densities.reshape(-1)

# %%
USE_PLOTLY = False
if USE_PLOTLY:
    import plotly.express as px

    fig = px.histogram(
        x=densities,
        log_y=True,
        title="Densities",
        labels={"x": "Values", "y": "Frequency"},
    )

    fig.show()
else:
    plt.hist(densities, log=True)
    plt.title("Densities")
    plt.show()

# %%
threshold = 250
filtered_points = points[densities > threshold]
filtered_densities = densities[densities > threshold]
colored_points_scatter = get_colored_points_scatter(
    points=filtered_points, colors=filtered_densities, offset=obj_offset
)

# Add the scatter plot to a figure and display it
fig = plot_obj(obj_filepath, scale=mesh_scale, offset=obj_offset)
fig.add_trace(colored_points_scatter)
fig.update_layout(legend_orientation="h")

fig.show()

# %%
mesh.centroid * mesh_scale, mesh.extents * mesh_scale, mesh.bounds * mesh_scale

# %% [markdown]
# # Create Dataset

# %% [markdown]
# # Visualize Data

# %% [markdown]
# # Create Model

# %% [markdown]
# # Run Training

# %% [markdown]
# # Run Evaluation

# %% [markdown]
# # Visualize Results

# %%
