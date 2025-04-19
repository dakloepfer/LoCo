"""A script to generate better depth maps than by inpainting; this first uses the 3D meshes to fill in missing values, and only then uses inpainting to fill in the rest."""

import argparse
import os
import warnings
import zipfile

import cv2
import numpy as np
import open3d
from einops import rearrange, repeat
from PIL import Image
from scipy import sparse
from tqdm import tqdm

# from natsort import natsorted


def invert_se3(se3_matrix):
    """Invert an SE(3) transformations.

    Parameters
    ----------
    se3_matrix (4 x 4 tensor)
        The SE(3) transformations to invert.

    Returns
    -------
    4 x 4 tensor
        The inverted transformation.
    """

    R = se3_matrix[:3, :3]
    t = se3_matrix[:3, 3:]

    inverse = np.eye(4, dtype=se3_matrix.dtype)

    inverse_R = R.T
    inverse[:3, :3] = inverse_R
    inverse[:3, 3:] = -inverse_R @ t

    return inverse


def bad_depth_values(depth, min_depth=0.01, max_depth=20):
    return (depth < min_depth) | (depth > max_depth) | np.isnan(depth) | np.isinf(depth)


def inpaint_depth(depth, image, nyu_depth_dataset=False, alpha=1):
    """In-paint depth values using method from "Colorization Using Optimization" by Levin et al. (https://www.cs.huji.ac.il/w~yweiss/Colorization/), which was also used / adapted for depth inpainting in the NYU Depth Dataset (https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).

    Parameters
    ----------
    depth (height x width numpy array):
        the depth values with some missing / noisy values

    image (height x width x channels numpy array):
        the RGB image, whose intensities are used as a guide for the inpainting

    nyu_depth_dataset (bool, optional):
        Whether I should use the method from the NYU Depth Dataset code or the one from the original paper, by default False

    alpha (int, optional):
        Only used if nyu_depth_dataset=True, this is "a penalty value betwwen 0 and 1 for the current depth values", by default 1

    Returns
    -------
    height x width numpy array:
        the inpainted depth map
    """
    window_radius = 1
    window_size = (2 * window_radius + 1) ** 2

    height, width = depth.shape
    n_pixels = height * width

    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).squeeze().astype(np.float32)

    depth_is_noise = bad_depth_values(depth)

    if depth_is_noise.sum() == 0:
        return depth

    if depth_is_noise.sum() >= 0.5 * n_pixels:
        warnings.warn("Not enough valid depth values to inpaint")
        return depth

    # normalise depth
    max_depth = np.max(depth[~depth_is_noise])
    depth = depth / max_depth
    depth[depth > 1] = 1

    row_inds = []  # the row indices for the sparse (1 - weights) matrix
    col_inds = []  # the col indices for the sparse (1 - weights) matrix
    vals = []  # the values for the sparse (1 - weights) matrix

    grayscale = rearrange(grayscale, "h w -> (h w)")

    all_pixel_indices = np.arange(n_pixels).reshape(height, width)
    noise_pixel_indices = all_pixel_indices[depth_is_noise]
    n_noise_pixels = len(noise_pixel_indices)

    if nyu_depth_dataset:
        # diagonal for non-noise pixels is alpha, not 1
        non_noise_pixel_idxs = all_pixel_indices[~depth_is_noise]
        row_inds.append(non_noise_pixel_idxs)
        col_inds.append(non_noise_pixel_idxs)
        vals.append(alpha * np.ones(len(non_noise_pixel_idxs)))

        # diagonal for noise pixels is still 1
        row_inds.append(noise_pixel_indices)
        col_inds.append(noise_pixel_indices)
        vals.append(np.ones(n_noise_pixels, dtype=np.float32))
    else:
        row_inds.append(np.arange(n_pixels))
        col_inds.append(np.arange(n_pixels))
        vals.append(np.ones(n_pixels, dtype=np.float32))

    # For the noise pixels, the diagonal is again 1 (which I have already dealt with), but the weighting values for the other pixels in the noise pixel's window are calculated using the weighting function
    noise_pixels = np.argwhere(depth_is_noise)

    window_indices = np.zeros((n_noise_pixels, window_size), dtype=int)

    window_indices[:, 0] = noise_pixel_indices  # central
    window_indices[:, 1] = noise_pixel_indices - width - 1  # upper left
    window_indices[:, 2] = noise_pixel_indices - width  # upper middle
    window_indices[:, 3] = noise_pixel_indices - width + 1  # upper right
    window_indices[:, 4] = noise_pixel_indices - 1  # left
    window_indices[:, 5] = noise_pixel_indices + 1  # right
    window_indices[:, 6] = noise_pixel_indices + width - 1  # lower left
    window_indices[:, 7] = noise_pixel_indices + width  # lower middle
    window_indices[:, 8] = noise_pixel_indices + width + 1  # lower right

    valid_window_indices_mask = np.ones_like(window_indices, dtype=bool)
    valid_window_indices_mask[:, 1] = (noise_pixels[:, 0] >= 1) & (
        noise_pixels[:, 1] >= 1
    )
    valid_window_indices_mask[:, 2] = noise_pixels[:, 0] >= 1
    valid_window_indices_mask[:, 3] = (noise_pixels[:, 0] >= 1) & (
        noise_pixels[:, 1] <= width - 2
    )
    valid_window_indices_mask[:, 4] = noise_pixels[:, 1] >= 1
    valid_window_indices_mask[:, 5] = noise_pixels[:, 1] <= width - 2
    valid_window_indices_mask[:, 6] = (noise_pixels[:, 0] <= height - 2) & (
        noise_pixels[:, 1] >= 1
    )
    valid_window_indices_mask[:, 7] = noise_pixels[:, 0] <= height - 2
    valid_window_indices_mask[:, 8] = (noise_pixels[:, 0] <= height - 2) & (
        noise_pixels[:, 1] <= width - 2
    )

    window_indices[~valid_window_indices_mask] = 0

    window_weight_vals = grayscale[window_indices]
    window_gs_variance = np.var(
        window_weight_vals, axis=1, where=valid_window_indices_mask
    )

    # from now mask out the central pixel, was only needed for variance
    central_pixel_val = window_weight_vals[:, 0]
    window_indices = window_indices[:, 1:]
    window_weight_vals = window_weight_vals[:, 1:]
    valid_window_indices_mask = valid_window_indices_mask[:, 1:]

    csig = 0.6 * window_gs_variance
    mgv = np.min(
        (window_weight_vals - central_pixel_val[:, None]) ** 2,
        axis=1,
        where=valid_window_indices_mask,
        initial=np.inf,
    )

    csig = np.clip(csig, a_min=-mgv / np.log(0.01), a_max=None)
    csig = np.clip(csig, a_min=2.2e-6, a_max=None)

    window_weight_vals = np.exp(
        -((window_weight_vals - central_pixel_val[:, None]) ** 2) / csig[:, None]
    )
    window_weight_vals /= np.sum(
        window_weight_vals, axis=1, where=valid_window_indices_mask, keepdims=True
    )

    # the -weights part in the (1-weights) matrix
    row_inds.append(
        repeat(
            noise_pixel_indices,
            "n_noise_pixels -> n_noise_pixels window_size",
            n_noise_pixels=n_noise_pixels,
            window_size=window_size - 1,
        )[valid_window_indices_mask]
    )
    col_inds.append(window_indices[valid_window_indices_mask])
    vals.append(-window_weight_vals[valid_window_indices_mask])

    row_inds = np.concatenate(row_inds)
    col_inds = np.concatenate(col_inds)
    vals = np.concatenate(vals)

    # NOTE: I would have thought that the equation should be slightly different, namely if the weights matrix is filled properly for all, then for the depth_is_noise values (1 - weights.T) @ (1 - weights) @ new_depths = 0, (with for the non-noise values still new_depths = old depths)
    # Here I solve (1 - weights) @ new_depths = 0 (with for the non-noise values still new_depths = old depths), any solution of which also solves the above equation, but maybe there is no solution to this equation but to the correct one. If I run into any errors to that effect, I might want to change this. Since this is a linear system of equations which seem pretty well-behaved, I don't expect something like this though.

    A = sparse.coo_matrix((vals, (row_inds, col_inds)), shape=(n_pixels, n_pixels))
    b = np.zeros(n_pixels)
    b[(~depth_is_noise).flatten()] = depth[~depth_is_noise].flatten()

    if nyu_depth_dataset:
        # This way of handling alpha is not what the code provided by the NYU Depths authors is doing, but what they are doing seems very weird; I think their code is missing a line to skip non-noise pixels in the loop above
        # If I am correct, then I think alpha=1 is simply the original code, while alpha=0 allows the known depths to be ignored completely
        b *= alpha

    new_depths = sparse.linalg.spsolve(A.tocsc(), b)

    return rearrange(new_depths * max_depth, "(h w) -> h w", h=height, w=width)


def calc_depths_from_raycaster(
    raycaster, camera_pose, intrinsic_matrix, bad_depths, opts
):
    """Calculate the depths at the pixels with bad depth values using the 3D mesh.

    Parameters
    ----------
    raycaster (open3d.t.geometry.RaycastingScene):
        The raycaster for the 3D mesh

    camera_pose (4x4 numpy array):
        The camera-to-world transformation matrix.

    intrinsic_matrix (3x3 numpy array):
        The intrinsic camera matrix.

    bad_depths (height x width bool numpy array):
        A boolean array indicating which pixels have bad depth values. Function will calculate the depth values for the pixels for which bad_depths=True.

    opts (Namespace):
        Command line options

    Returns
    -------
    depth_values (n_bad_depths numpy array):
        The depth values for the pixels with bad depth values.

    """
    # the camera pose is the camera-to-world transformation, so invert_se3(camera_pose) is the world-to-camera transformation.
    extrinsic_matrix = invert_se3(camera_pose)
    extrinsic_matrix = open3d.core.Tensor(extrinsic_matrix)

    px = round(intrinsic_matrix[0, 2])
    py = round(intrinsic_matrix[1, 2])
    intrinsic_matrix = open3d.core.Tensor(intrinsic_matrix)

    rays = open3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix, extrinsic_matrix, opts.img_width, opts.img_height
    ).numpy()

    # normalise ray direction vectors to unit length
    rays[:, :, 3:] = rays[:, :, 3:] / np.linalg.norm(
        rays[:, :, 3:], axis=-1, keepdims=True
    )

    ray_directions = rays[:, :, 3:]
    center_ray_direction = ray_directions[py, px]

    # only compute depths for pixels with bad depth values
    ray_directions = ray_directions[bad_depths]
    rays = rays[bad_depths]

    tensor_rays = open3d.core.Tensor(rays)

    distances_to_hit = raycaster.cast_rays(tensor_rays)["t_hit"].numpy()
    depth_values = distances_to_hit * np.einsum(
        "i,ni->n", center_ray_direction, ray_directions
    )  # project the ray direction onto the center ray direction to get the depth values

    # set depth values to 0 for pixels that are not hit
    depth_values[depth_values > 100] = 0

    return depth_values


def load_raycaster(scene_dir):

    scene_name = os.path.basename(scene_dir)
    mesh_file = os.path.join(scene_dir, f"{scene_name}_vh_clean_2.ply")
    if not os.path.exists(mesh_file):
        return None

    mesh = open3d.io.read_triangle_mesh(mesh_file)
    mesh_triangles = np.asarray(mesh.triangles).astype(np.float32)
    mesh_vertices = np.asarray(mesh.vertices).astype(np.float32)

    # For some reason open3d has more than one TriangleMesh shape
    raycaster_mesh = open3d.t.geometry.TriangleMesh(
        open3d.core.Tensor(mesh_vertices),
        open3d.core.Tensor(mesh_triangles),
    )

    raycaster = open3d.t.geometry.RaycastingScene()
    raycaster.add_triangles(raycaster_mesh)

    return raycaster


def precompute_color_and_depth(scene, scene_dir, opts):

    old_color_imgs_path = os.path.join(scene_dir, "color")
    old_depth_path = os.path.join(scene_dir, "depth")
    cam_pose_path = os.path.join(scene_dir, "pose")

    if not os.path.exists(os.path.join(scene_dir, opts.color_dir_name)):
        os.makedirs(os.path.join(scene_dir, opts.color_dir_name))
    if not os.path.exists(os.path.join(scene_dir, opts.depth_dir_name)):
        os.makedirs(os.path.join(scene_dir, opts.depth_dir_name))

    # load camera parameters for depths to fill in bad depths
    intrinsic_matrix = np.loadtxt(
        os.path.join(scene_dir, "intrinsic", "intrinsic_depth.txt")
    )
    intrinsic_matrix = intrinsic_matrix[:3, :3]

    # adjust for larger pixels
    intrinsic_matrix[0, 0] *= opts.img_width / 640
    intrinsic_matrix[1, 1] *= opts.img_height / 480
    intrinsic_matrix[0, 2] *= opts.img_width / 640
    intrinsic_matrix[1, 2] *= opts.img_height / 480

    # Load raycaster for scene
    raycaster = load_raycaster(scene_dir)

    n_times_meshes_were_used = 0
    n_times_inpainting_was_used = 0
    n_images_skipped = 0
    fraction_filled_by_mesh = []

    n_images = len(os.listdir(old_color_imgs_path))

    for i in tqdm(
        range(n_images),
        desc="Precomputing images and depths for scene {}".format(scene),
    ):
        camera_pose = np.loadtxt(os.path.join(cam_pose_path, f"{i}.txt"))
        if np.isinf(camera_pose).any() and not opts.save_all:
            print(f"Skipping image {i} in scene {scene} due to invalid camera pose")
            n_images_skipped += 1
            continue

        image_file_name = f"{i}.jpg"
        depth_file_name = f"{i}.png"

        # Load ground-truth color image
        img = cv2.imread(os.path.join(old_color_imgs_path, image_file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (opts.img_width, opts.img_height))

        # Load ground-truth depth image
        depths = cv2.imread(
            os.path.join(old_depth_path, depth_file_name), cv2.IMREAD_ANYDEPTH
        ).astype(np.float32)
        depths = depths / 1000.0  # scale to metres
        depths = cv2.resize(
            depths,
            (opts.img_width, opts.img_height),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )

        # fill in bad or missing depth values
        bad_depths = bad_depth_values(depths)
        n_bad_gt_depths = np.sum(bad_depths)

        if n_bad_gt_depths > 0:  # fill in bad depth values using 3D mesh
            if raycaster is not None:
                bad_depths_from_mesh = calc_depths_from_raycaster(
                    raycaster, camera_pose, intrinsic_matrix, bad_depths, opts
                )
                depths[bad_depths] = bad_depths_from_mesh

            bad_depths = bad_depth_values(depths)
            n_bad_depths_v1 = np.sum(bad_depths)
            if opts.verbose:
                print(
                    "Reduced number of bad depths from {} to {} using the 3D Mesh".format(
                        n_bad_gt_depths, n_bad_depths_v1
                    )
                )
            fraction_filled_by_mesh.append(
                (n_bad_gt_depths - n_bad_depths_v1) / n_bad_gt_depths
            )

            n_times_meshes_were_used += 1
            if (
                n_bad_depths_v1 >= 0.2 * opts.img_width * opts.img_height
                and not opts.save_all
            ):
                n_images_skipped += 1
                # too many bad depths to fill in with inpainting
                print(
                    "Skipping image {} in scene {} due to too many bad depths".format(
                        image_file_name, scene
                    )
                )
                continue

            elif n_bad_depths_v1 > 0:  # fill in bad depth values using inpainting
                n_times_inpainting_was_used += 1

                depths = inpaint_depth(depths, img)

                bad_depths = bad_depth_values(depths)
                n_bad_depths_final = np.sum(bad_depths)
                if opts.verbose:
                    print(
                        "Reduced number of bad depths from {} to {} using inpainting".format(
                            n_bad_depths_v1, n_bad_depths_final
                        )
                    )

        if not opts.save_all:
            assert np.sum(bad_depth_values(depths)) == 0

        np.save(os.path.join(scene_dir, opts.color_dir_name, f"{i}"), img)
        np.save(os.path.join(scene_dir, opts.depth_dir_name, f"{i}"), depths)

    print("Finished precomputing scene {}".format(scene))
    print(
        f"Meshes were used to fill in bad depths {n_times_meshes_were_used} times, on average filling in a fraction of {100 * sum(fraction_filled_by_mesh) / len(fraction_filled_by_mesh):.3f}% of the bad depths"
    )
    print(
        "Inpainting was used to fill in bad depths {} times".format(
            n_times_inpainting_was_used
        )
    )
    print("{} images were skipped due to too many bad depths".format(n_images_skipped))


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

        precompute_color_and_depth(scene, scene_dir, opts)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--raw_dataset_dir",
        type=str,
        default="data/scannet/scans",
    )
    parser.add_argument("--img_width", type=int, default=320)
    parser.add_argument("--img_height", type=int, default=240)
    parser.add_argument("--color_dir_name", type=str, default="color_npy")
    parser.add_argument("--depth_dir_name", type=str, default="depth_npy")
    parser.add_argument("--start_scene", type=str, default=None)
    parser.add_argument("--end_scene", type=str, default=None)
    parser.add_argument(
        "--save_all",
        action="store_true",
        help="Whether to save all images and depths, including ones with unreliable depths or invalid poses",
    )
    opts = parser.parse_args()
    main(opts)
