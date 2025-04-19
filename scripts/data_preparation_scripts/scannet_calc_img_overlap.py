import argparse
import os
import warnings

import numpy as np
import torch
from einops import einsum, rearrange, repeat
from kornia.geometry.camera import PinholeCamera
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


def invert_intrinsic_matrix(intrinsic_matrices):
    """Invert a batch of intrinsic matrices with zero skew.

    Parameters
    ----------
    intrinsic_matrices (batch_size x 3 x 3 tensor):
        The intrinsic matrices to invert

    Returns
    -------
    batch_size x 3 x 3 tensor:
        The inverted intrinsic matrices.
    """

    inverse = (
        torch.eye(3, device=intrinsic_matrices.device)
        .unsqueeze(dim=0)
        .repeat(intrinsic_matrices.shape[0], 1, 1)
    )
    inverse[:, 0, 0] = 1 / intrinsic_matrices[:, 0, 0]
    inverse[:, 1, 1] = 1 / intrinsic_matrices[:, 1, 1]
    inverse[:, 0, 2] = -intrinsic_matrices[:, 0, 2] / intrinsic_matrices[:, 0, 0]
    inverse[:, 1, 2] = -intrinsic_matrices[:, 1, 2] / intrinsic_matrices[:, 1, 1]

    return inverse


def calculate_pairwise_overlaps(
    depths, cam_to_world_matrices, intrinsic_matrices, patch_locations
):
    """Calculate the overlaps between each pair of the images defined.

    Parameters
    ----------
    depths (n_imgs x height x width tensor):
        the depth maps in metres for the images.

    cam_to_world_matrices (n_imgs x 4 x 4 tensor):
        The camera to world matrices for the images.

    intrinsic_matrices (n_imgs x 3 x 3 tensor):
        The intrinsic matrices for the images.

    patch_locations (n_imgs x height x width x 3 tensor):
        tensor containing the world-coordinates of the patch centers for each image.

    Returns
    -------
    n_imgs x n_imgs tensor:
        The overlaps between each pair of images. Element (i, j) is the fraction of pixels in image i that are visible (i.e. in the image and not occluded) in image j.
    """
    n_imgs, height, width = depths.shape
    assert patch_locations.shape[:-1] == (n_imgs, height, width)

    overlaps = torch.eye(n_imgs, device=depths.device)  # the diagonals will be 1.0
    if intrinsic_matrices.shape[1] == 3:
        intrinsics = torch.eye(4).repeat(n_imgs, 1, 1).to(intrinsic_matrices.device)
        intrinsics[:, :3, :3] = intrinsic_matrices
    else:
        intrinsics = intrinsic_matrices

    world_to_cam_matrices = invert_se3(cam_to_world_matrices)
    camera = PinholeCamera(
        intrinsics,
        world_to_cam_matrices,
        torch.ones_like(depths[:, 0, 0]) * height,
        torch.ones_like(depths[:, 0, 0]) * width,
    )
    depths = rearrange(depths, "b h w -> b (h w)")

    for i in tqdm(range(n_imgs), desc="Calculating overlaps"):

        pixel_coordinates = camera.project(
            repeat(patch_locations[i], "h w c -> b (h w) c", b=n_imgs)
        )  # b (h w) 2
        pixel_coordinates = rearrange(
            pixel_coordinates, "b (h w) t -> b h w t", h=height, w=width
        )
        depths_in_all_imgs = einsum(
            camera.extrinsics,
            torch.cat(
                [
                    patch_locations[i],
                    torch.ones((height, width, 1), device=depths.device),
                ],
                dim=-1,
            ),
            "b f f2, h w f2 -> b h w f",
        )[
            :, :, :, 2
        ]  # b (h w)

        # Check visibility
        rounded_pixel_idxs = rearrange(
            torch.clamp(
                torch.round(pixel_coordinates[:, :, :, 0]), min=0, max=width - 1
            )
            + torch.clamp(
                torch.round(pixel_coordinates[:, :, :, 1]), min=0, max=height - 1
            )
            * width,
            "b h w -> b (h w)",
        ).to(torch.int64)

        depths_at_pixel_coordinates = torch.gather(
            depths, dim=1, index=rounded_pixel_idxs
        )
        depths_at_pixel_coordinates = rearrange(
            depths_at_pixel_coordinates, "b (h w) -> b h w", h=height, w=width
        )
        visible = (
            (pixel_coordinates[:, :, :, 0] >= 0)
            & (pixel_coordinates[:, :, :, 0] < width)
            & (pixel_coordinates[:, :, :, 1] >= 0)
            & (pixel_coordinates[:, :, :, 1] < height)
            & (depths_in_all_imgs > 0)
            & (depths_in_all_imgs < depths_at_pixel_coordinates + 5e-2)
        )

        overlaps[i] = torch.sum(visible, dim=(1, 2)) / (height * width)

    return overlaps


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
        assert not (
            np.isinf(camera_to_world_matrices[-1]).any()
            or np.isnan(camera_to_world_matrices[-1]).any()
        )

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

    depths = torch.tensor(depths, dtype=torch.float, device=opts.device)
    intrinsic_matrix = torch.tensor(
        intrinsic_matrix, dtype=torch.float, device=opts.device
    )
    camera_to_world_matrices = torch.tensor(
        camera_to_world_matrices, dtype=torch.float, device=opts.device
    )

    assert depths.shape[0] == camera_to_world_matrices.shape[0]
    assert depths.shape[0] == len(depth_files)
    return depths, camera_to_world_matrices, intrinsic_matrix


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
        print(f"Processing scene {scene}...")

        scene_dir = os.path.join(opts.raw_dataset_dir, scene)

        depths, cam_to_world_matrices, intrinsic_matrix = load_scene_information(
            opts, scene
        )
        patch_locations = torch.from_numpy(
            np.load(os.path.join(scene_dir, opts.patch_location_file))
        )

        patch_locations = torch.from_numpy(
            np.load(os.path.join(scene_dir, opts.patch_location_file))
        ).to(opts.device)
        scene_overlaps = calculate_pairwise_overlaps(
            depths, cam_to_world_matrices, intrinsic_matrix, patch_locations
        )

        np.save(
            os.path.join(scene_dir, "img_overlaps.npy"), scene_overlaps.cpu().numpy()
        )


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
        default=30,
        help="Image height to use for calculations",
    )
    parser.add_argument(
        "--img_width", type=int, default=40, help="Image width to use for calculations"
    )
    parser.add_argument(
        "--patch_location_file", type=str, default="patch_locations_30_40.npy"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for calculations"
    )
    parser.add_argument("--start_scene", type=str, default=None)
    parser.add_argument("--end_scene", type=str, default=None)
    opts = parser.parse_args()
    opts.device = torch.device(opts.device)
    main(opts)
