"""Calculate the 3D locations for (the center of) every pixel patch"""

# NOTE: I could also integrate this with matterport3d_calc_img_overlap.py, but I feel keeping them separate makes it easier to update them individually.
import argparse
import os
import warnings
from math import ceil

import numpy as np
import torch
from einops import einsum, rearrange, reduce, repeat
from kornia.geometry.camera import PinholeCamera
from kornia.utils import create_meshgrid
from tqdm import tqdm

ALL_SCENES = [
    "17DRP5sb8fy",
    "1LXtFkjw3qL",
    "1pXnuDYAj8r",
    "29hnd4uzFmX",
    "2azQ1b91cZZ",
    "2n8kARJN3HM",
    "2t7WUuJeko7",
    "5LpN3gDmAk7",
    "5q7pvUzZiYa",
    "5ZKStnWn8Zo",
    "759xd9YjKW5",
    "7y3sRwLe3Va",
    "8194nk5LbLH",
    "82sE5b5pLXE",
    "8WUmhLawc2A",
    "aayBHfsNo7d",
    "ac26ZMwG7aT",
    "ARNzJeq3xxb",
    "B6ByNegPMKs",
    "b8cTxDM8gDG",
    "cV4RVeZvu5T",
    "D7G3Y4RVNrH",
    "D7N2EKCX4Sj",
    "dhjEzFoUFzH",
    "E9uDoFAP3SH",
    "e9zR4mvMWw7",
    "EDJbREhghzL",
    "EU6Fwq7SyZv",
    "fzynW3qQPVF",
    "GdvgFV5R1Z5",
    "gTV8FGcVJC9",
    "gxdoqLR6rwA",
    "gYvKGZ5eRqb",
    "gZ6f7yhEvPG",
    "HxpKQynjfin",
    "i5noydFURQK",
    "JeFG25nYj2p",
    "JF19kD82Mey",
    "jh4fc5c5qoQ",
    "JmbYfDe2QKZ",
    "jtcxE69GiFV",
    "kEZ7cmS4wCh",
    "mJXqzFtmKg4",
    "oLBMNvg9in8",
    "p5wJjkQkbXX",
    "pa4otMbVnkk",
    "pLe4wQe7qrG",
    "Pm6F8kyY3z2",
    "pRbA3pwrgk9",
    "PuKPg4mmafe",
    "PX4nDJXEHrG",
    "q9vSo1VnCiC",
    "qoiz87JEwZ2",
    "QUCTc6BB5sX",
    "r1Q1Z4BcV1o",
    "r47D5H71a5s",
    "rPc6DW4iMge",
    "RPmz2sHmrrY",
    "rqfALeAoiTq",
    "s8pcmisQ38h",
    "S9hNv5qa7GM",
    "sKLMLpTHeUy",
    "SN83YJsR3w2",
    "sT4fr6TAbpF",
    "TbHJrupSAjP",
    "ULsKaCPVFJR",
    "uNb9QFRL6hY",
    "ur6pFq6Qu1A",
    "UwV83HsGsw3",
    "Uxmj2M2itWa",
    "V2XKFyX4ASd",
    "VFuaQ6m2Qom",
    "VLzqgDo317F",
    "Vt2qJdWjCF2",
    "VVfe2KiqLaN",
    "Vvot9Ly1tCj",
    "vyrNrziPKCB",
    "VzqfbhrpDEA",
    "wc2JMjhGNzB",
    "WYY7iVyf5p8",
    "X7HyMhZNoso",
    "x8F5xyUWy9e",
    "XcA2TqTSSAj",
    "YFuZgdQ5vWj",
    "YmJkqBEsHnH",
    "yqstnuAEVhm",
    "YVUC4YcDtcY",
    "Z6MFQCViBuw",
    "ZMojNkEp431",
    "zsNo4HB9uLZ",
]

TRAIN_SCENES = [
    "r47D5H71a5s",
    "sKLMLpTHeUy",
    "VFuaQ6m2Qom",
    "sT4fr6TAbpF",
    "gTV8FGcVJC9",
    "VVfe2KiqLaN",
    "XcA2TqTSSAj",
    "Vvot9Ly1tCj",
    "E9uDoFAP3SH",
    "5LpN3gDmAk7",
    "JF19kD82Mey",
    "uNb9QFRL6hY",
    "VLzqgDo317F",
    "ZMojNkEp431",
    # ---
    "s8pcmisQ38h",
    "1LXtFkjw3qL",
    "PX4nDJXEHrG",
    "mJXqzFtmKg4",
    "SN83YJsR3w2",
    "kEZ7cmS4wCh",
    "8WUmhLawc2A",
    "e9zR4mvMWw7",
    "qoiz87JEwZ2",
    "759xd9YjKW5",
    "7y3sRwLe3Va",
    "vyrNrziPKCB",
    "aayBHfsNo7d",
    "b8cTxDM8gDG",
    "ur6pFq6Qu1A",
    # # ---
    "29hnd4uzFmX",
    "i5noydFURQK",
    "dhjEzFoUFzH",
    "D7G3Y4RVNrH",
    "D7N2EKCX4Sj",
    "S9hNv5qa7GM",
    "r1Q1Z4BcV1o",
    "rPc6DW4iMge",
    "gZ6f7yhEvPG",
    "ac26ZMwG7aT",
    "17DRP5sb8fy",
    "82sE5b5pLXE",
    "Pm6F8kyY3z2",
    "ULsKaCPVFJR",
    "Uxmj2M2itWa",
    # # ---
    "JeFG25nYj2p",
    "V2XKFyX4ASd",
    "YmJkqBEsHnH",
    "1pXnuDYAj8r",
    "EDJbREhghzL",
    "p5wJjkQkbXX",
    "pRbA3pwrgk9",
    "jh4fc5c5qoQ",
    "VzqfbhrpDEA",
    "B6ByNegPMKs",
    "JmbYfDe2QKZ",
    "2n8kARJN3HM",
    "PuKPg4mmafe",
    "cV4RVeZvu5T",
    "5q7pvUzZiYa",
]


def invert_se3(se3_matrix):
    """Invert a batch of SE(3) transformations.

    Parameters
    ----------
    se3_matrix (batch_size x 4 x 4 tensor)
        The SE(3) transformations to invert.

    Returns
    -------
    batch_size x 4 x 4 tensor
        The inverted transformations.
    """

    R = se3_matrix[:, :3, :3]
    t = se3_matrix[:, :3, 3:]

    inverse = (
        torch.eye(4, device=se3_matrix.device)
        .unsqueeze(dim=0)
        .repeat(se3_matrix.shape[0], 1, 1)
    )

    inverse_R = rearrange(R, "b i j -> b j i")
    inverse[:, :3, :3] = inverse_R
    inverse[:, :3, 3:] = -inverse_R @ t

    return inverse


def compute_patch_locations(
    depths, cam_to_world_matrices, intrinsic_matrices, patch_size
):
    """Calculate the 3D locations of the (centers of) each pixel patch for the images provided.

    Parameters
    ----------
    depths (n_imgs x height x width tensor):
        the depth map in metres for the images.

    cam_to_world_matrices (n_imgs x 4 x 4 tensor):
        The camera to world matrices for the images

    intrinsic_matrices (n_imgs x 3 x 3 tensor):
        The intrinsic matrices for the images

    patch_size (int):
        The size of the patches to use.

    Returns
    -------
    n_imgs x (height // patch_size) x (width // patch_size) x 3 tensor:
        The 3D locations (x, y, z) of the (centers of) each pixel patch for the images provided.
    """
    device = depths.device
    n_imgs, height, width = depths.shape
    world_to_cam_matrices = invert_se3(cam_to_world_matrices)
    intrinsics = torch.zeros((n_imgs, 4, 4), device=device)
    intrinsics[:, :3, :3] = intrinsic_matrices
    intrinsics[:, 3, 3] = 1.0

    camera = PinholeCamera(
        intrinsics,
        world_to_cam_matrices,
        torch.ones_like(depths[:, 0, 0]) * height,
        torch.ones_like(depths[:, 0, 0]) * width,
    )

    patch_pixel_coords = (
        create_meshgrid(
            height // patch_size,
            width // patch_size,
            normalized_coordinates=False,
            device=device,
        )
        + 0.5
    ) * patch_size
    patch_pixel_coords = repeat(patch_pixel_coords, "() h w c -> n h w c", n=n_imgs)

    downsampled_depths = torch.nn.functional.interpolate(
        depths.unsqueeze(1), scale_factor=1 / patch_size, mode="nearest-exact"
    )
    downsampled_depths = rearrange(
        downsampled_depths,
        "n () h w -> n h w ()",
        n=n_imgs,
        h=height // patch_size,
        w=width // patch_size,
    )

    patch_3d_locations = camera.unproject(patch_pixel_coords, downsampled_depths)

    return patch_3d_locations


def load_scene_information(opts, scene):
    scene_dir = os.path.join(opts.raw_dataset_dir, scene)
    depth_dir = os.path.join(scene_dir, "depth")

    depth_files = sorted(os.listdir(depth_dir))

    # Load depth images
    depths = []
    for depth_file in tqdm(depth_files, "Loading depth images"):
        depths.append(np.load(os.path.join(depth_dir, depth_file)))

    depths = np.stack(depths, axis=0)
    if depths.shape[1:] != (opts.img_height, opts.img_width):
        depths = torch.tensor(depths).unsqueeze(dim=1)
        depths = (
            torch.nn.functional.interpolate(
                depths, size=(opts.img_height, opts.img_width), mode="nearest-exact"
            )
            .squeeze(dim=1)
            .numpy()
        )
        warnings.warn(
            f"Resized depth images to shape ({opts.img_height}, {opts.img_width})"
        )

    # Load camera matrices
    camera_parameter_file_path = os.path.join(
        scene_dir, "undistorted_camera_parameters", "{}.conf".format(scene)
    )
    with open(camera_parameter_file_path, "r") as f:
        parameter_file_lines = f.readlines()

    current_intrinsics = None
    intrinsic_matrices = np.zeros((len(depth_files), 3, 3))
    cam_to_world_matrices = np.zeros((len(depth_files), 4, 4))

    for line in tqdm(parameter_file_lines, "Loading camera parameters"):
        if line.startswith("intrinsics"):
            current_intrinsics = rearrange(
                np.array(
                    [
                        float(x)
                        for x in line.strip().split(" ")[1:]
                        if not (x.isspace() or len(x) == 0)
                    ]
                ),
                "(h w) -> h w",
                h=3,
                w=3,
            )
            current_intrinsics[:2, :] *= opts.img_height / 1024.0
            # take into account that Matterport3D has origin in bottom left, not top left, with y-axis pointing up
            current_intrinsics[1, 2] = opts.img_height - current_intrinsics[1, 2]

        elif line.startswith("scan"):
            scan_line = line.strip().split(" ")[1:]
            depth_file_name = scan_line[0].replace(".png", ".npy")

            if depth_file_name not in depth_files:
                continue

            depth_file_index = depth_files.index(depth_file_name)

            if intrinsic_matrices[depth_file_index].any():
                raise ValueError(
                    "Intrinsics already set for this image {}".format(depth_file_name)
                )
            if cam_to_world_matrices[depth_file_index].any():
                raise ValueError(
                    "Cam to world matrix already set for this image {}".format(
                        depth_file_name
                    )
                )

            intrinsic_matrices[depth_file_index] = current_intrinsics
            cam_to_world_pose = rearrange(
                np.array([float(x) for x in scan_line[2:]]),
                "(h w) -> h w",
                h=4,
                w=4,
            )
            # Matterport3D camera coordinate system is z into camera, x right, y up; need to switch to x right, y down, z out of camera
            cam_to_world_pose = einsum(
                cam_to_world_pose,
                torch.tensor(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                    dtype=torch.float,
                ),
                "i j, j k -> i k",
            )
            cam_to_world_matrices[depth_file_index] = cam_to_world_pose

        else:
            continue

    depths = torch.tensor(depths, dtype=torch.float)
    intrinsic_matrices = torch.tensor(intrinsic_matrices, dtype=torch.float)
    cam_to_world_matrices = torch.tensor(cam_to_world_matrices, dtype=torch.float)
    assert depths.shape[0] == intrinsic_matrices.shape[0]
    assert depths.shape[0] == cam_to_world_matrices.shape[0]
    assert depths.shape[0] == len(depth_files)
    return depths, cam_to_world_matrices, intrinsic_matrices, depth_files


def main(opts):
    for scene in ALL_SCENES:
        print(f"Processing scene {scene}...")

        scene_dir = os.path.join(opts.raw_dataset_dir, scene)
        patch_location_dir = os.path.join(
            scene_dir,
            f"patch_locations_{opts.img_height // opts.patch_size}_{opts.img_width // opts.patch_size}",
        )
        if not os.path.exists(patch_location_dir):
            os.makedirs(patch_location_dir)

        (
            depths,
            cam_to_world_matrices,
            intrinsic_matrices,
            depth_file_names,
        ) = load_scene_information(opts, scene)

        patch_location_file_names = [
            depth_file[:-8] + "pl" + depth_file[-7:] for depth_file in depth_file_names
        ]

        n_batches = ceil(len(depths) / opts.batch_size)
        all_locations = []
        for i in tqdm(range(n_batches), "Calculating patch locations"):
            batch_depths = depths[i * opts.batch_size : (i + 1) * opts.batch_size]
            batch_cam_to_world_matrices = cam_to_world_matrices[
                i * opts.batch_size : (i + 1) * opts.batch_size
            ]
            batch_intrinsic_matrices = intrinsic_matrices[
                i * opts.batch_size : (i + 1) * opts.batch_size
            ]
            batch_patch_locations = compute_patch_locations(
                batch_depths.to(opts.device),
                batch_cam_to_world_matrices.to(opts.device),
                batch_intrinsic_matrices.to(opts.device),
                opts.patch_size,
            )

            batch_location_file_names = patch_location_file_names[
                i * opts.batch_size : (i + 1) * opts.batch_size
            ]

            if opts.save_individually:
                for j, location_file_name in enumerate(batch_location_file_names):
                    location_file_path = os.path.join(
                        patch_location_dir, location_file_name
                    )

                    np.save(location_file_path, batch_patch_locations[j].cpu().numpy())
            else:
                if len(batch_patch_locations.shape) != 4:
                    batch_patch_locations = batch_patch_locations[None, ...]
                all_locations.append(batch_patch_locations.cpu().numpy())

        if not opts.save_individually:
            all_locations = np.concatenate(all_locations, axis=0)
            np.save(patch_location_dir, all_locations)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dataset_dir",
        type=str,
        default="data/matterport3d",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=256,
        help="Image height to use for calculations",
    )
    parser.add_argument(
        "--img_width", type=int, default=320, help="Image width to use for calculations"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for calculations"
    )
    parser.add_argument(
        "--patch_size", type=int, default=8, help="Patch size to use for calculations"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of images to calculate patch locations for in parallel",
    )
    parser.add_argument(
        "--save_individually",
        action="store_true",
        help="Save the patch locations for each image individually instead of in one big file.",
    )
    opts = parser.parse_args()
    opts.device = torch.device(opts.device)
    main(opts)
