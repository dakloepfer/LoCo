import argparse
import os

import numpy as np
import torch
from einops import einsum, rearrange, repeat
from kornia.geometry.camera import PinholeCamera
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


def main(opts):
    for scene in ALL_SCENES:
        print(f"Processing scene {scene}...")

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
                            for x in line.split(" ")[1:]
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
                scan_line = line.split(" ")[1:]
                depth_file_name = scan_line[0].replace(".png", ".npy")

                if depth_file_name not in depth_files:
                    continue

                depth_file_index = depth_files.index(depth_file_name)

                if intrinsic_matrices[depth_file_index].any():
                    raise ValueError(
                        "Intrinsics already set for this image {}".format(
                            depth_file_name
                        )
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

        depths = torch.tensor(depths, dtype=torch.float, device=opts.device)
        intrinsic_matrices = torch.tensor(
            intrinsic_matrices, dtype=torch.float, device=opts.device
        )
        cam_to_world_matrices = torch.tensor(
            cam_to_world_matrices, dtype=torch.float, device=opts.device
        )
        patch_locations = torch.from_numpy(
            np.load(os.path.join(scene_dir, opts.patch_location_file))
        ).to(opts.device)
        scene_overlaps = calculate_pairwise_overlaps(
            depths, cam_to_world_matrices, intrinsic_matrices, patch_locations
        )

        np.save(
            os.path.join(scene_dir, "img_overlaps.npy"), scene_overlaps.cpu().numpy()
        )


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
        default=32,
        help="Image height to use for calculations",
    )
    parser.add_argument(
        "--img_width", type=int, default=40, help="Image width to use for calculations"
    )
    parser.add_argument(
        "--patch_location_file", type=str, default="patch_locations_32_40.npy"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for calculations"
    )
    opts = parser.parse_args()
    opts.device = torch.device(opts.device)
    main(opts)
