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
from natsort import natsorted
from tqdm import tqdm


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
    depths, cam_to_world_matrices, intrinsic_matrix, patch_size
):
    """Calculate the 3D locations of the (centers of) each pixel patch for the images provided.

    Parameters
    ----------
    depths (n_imgs x height x width tensor):
        the depth map in metres for the images.

    cam_to_world_matrices (n_imgs x 4 x 4 tensor):
        The camera to world matrices for the images

    intrinsic_matrix (3 x 3 tensor):
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
    intrinsics[:, :3, :3] = repeat(intrinsic_matrix, "i j -> n i j", n=n_imgs)
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
    depth_dir = os.path.join(scene_dir, "depth_npy")
    pose_dir = os.path.join(scene_dir, "pose")

    depth_files = natsorted(os.listdir(depth_dir))

    # Load depth images
    depths = []
    camera_to_world_matrices = []
    for depth_file in tqdm(depth_files, "Loading depth images and camera poses"):
        depths.append(np.load(os.path.join(depth_dir, depth_file)))
        pose_file = depth_file.replace(".npy", ".txt")
        camera_to_world_matrices.append(np.loadtxt(os.path.join(pose_dir, pose_file)))

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

    camera_to_world_matrices = np.stack(camera_to_world_matrices, axis=0)
    assert camera_to_world_matrices.shape[1:] == (4, 4)

    # Load intrinsics
    intrinsic_matrix = np.loadtxt(
        os.path.join(scene_dir, "intrinsic", "intrinsic_color.txt")
    )
    intrinsic_matrix = intrinsic_matrix[:3, :3]

    # adjust for larger pixels
    intrinsic_matrix[0, 0] *= opts.img_width / 1296
    intrinsic_matrix[1, 1] *= opts.img_height / 968
    intrinsic_matrix[0, 2] *= opts.img_width / 1296
    intrinsic_matrix[1, 2] *= opts.img_height / 968

    depths = torch.tensor(depths, dtype=torch.float)
    intrinsic_matrix = torch.tensor(intrinsic_matrix, dtype=torch.float)
    camera_to_world_matrices = torch.tensor(camera_to_world_matrices, dtype=torch.float)

    assert depths.shape[0] == camera_to_world_matrices.shape[0]
    assert depths.shape[0] == len(depth_files)
    return depths, camera_to_world_matrices, intrinsic_matrix, depth_files


def main(opts):
    all_scenes = os.listdir(opts.raw_dataset_dir)
    all_scenes = sorted(all_scenes)
    all_scenes = [scene for scene in all_scenes if scene.startswith("scene")]

    if opts.start_scene is not None:
        try:
            start_idx = int(opts.start_scene)
        except ValueError:
            start_idx = all_scenes.index(opts.start_scene)
    else:
        start_idx = 0

    if opts.end_scene is not None:
        try:
            end_idx = int(opts.end_scene)
        except ValueError:
            end_idx = all_scenes.index(opts.end_scene)
    else:
        end_idx = len(all_scenes)
    all_scenes = all_scenes[start_idx:end_idx]

    for scene in all_scenes:
        print(f"\n =========================== \n Processing scene {scene}...\n")

        scene_dir = os.path.join(opts.raw_dataset_dir, scene)

        patch_location_dir = os.path.join(
            scene_dir,
            f"patch_locations_{opts.img_height // opts.patch_size}_{opts.img_width // opts.patch_size}",
        )
        if opts.save_individually:
            if not os.path.exists(patch_location_dir):
                os.makedirs(patch_location_dir)

        (
            depths,
            cam_to_world_matrices,
            intrinsic_matrix,
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

            batch_patch_locations = compute_patch_locations(
                batch_depths.to(opts.device),
                batch_cam_to_world_matrices.to(opts.device),
                intrinsic_matrix.to(opts.device),
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
        default="data/scannet/scans",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=240,
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
    parser.add_argument("--start_scene", type=str, default=None)
    parser.add_argument("--end_scene", type=str, default=None)
    opts = parser.parse_args()
    opts.device = torch.device(opts.device)
    main(opts)
