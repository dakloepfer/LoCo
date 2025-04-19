"""Use the triangle meshes and region_segmentations information to for each image give the semantic segmentation labels for each pixel."""

import argparse
import csv
import json
import os

import numpy as np
import open3d
from einops import rearrange
from PIL import Image
from tqdm import tqdm

# NOTE: had to clean up some of the labels; the only one that is maybe not completely obvious is replacing the "roof or floor / other room" with "ceiling"
MPCAT40_TO_INDEX = {
    "void": 0,
    "wall": 1,
    "floor": 2,
    "chair": 3,
    "door": 4,
    "table": 5,
    "picture": 6,
    "cabinet": 7,
    "cushion": 8,
    "window": 9,
    "sofa": 10,
    "bed": 11,
    "curtain": 12,
    "chest_of_drawers": 13,
    "plant": 14,
    "sink": 15,
    "stairs": 16,
    "ceiling": 17,
    "toilet": 18,
    "stool": 19,
    "towel": 20,
    "mirror": 21,
    "tv_monitor": 22,
    "shower": 23,
    "column": 24,
    "bathtub": 25,
    "counter": 26,
    "fireplace": 27,
    "lighting": 28,
    "beam": 29,
    "railing": 30,
    "shelving": 31,
    "blinds": 32,
    "gym_equipment": 33,
    "seating": 34,
    "board_panel": 35,
    "furniture": 36,
    "appliances": 37,
    "clothes": 38,
    "objects": 39,
    "misc": 40,
    "unlabeled": 41,
}
STUFF_CLASSES = {
    "void": 0,
    "wall": 1,
    "floor": 2,
    "curtain": 3,
    "stairs": 4,
    "ceiling": 5,
    "mirror": 6,
    "shower": 7,
    "column": 8,
    "beam": 9,
    "railing": 10,
    "shelving": 11,
    "blinds": 12,
    "board_panel": 13,
    "misc": 14,
    "unlabeled": 15,
}

STUFF_CLASSES_V1 = {
    "void": 0,
    "wall": 1,
    "floor": 2,
    "ceiling": 3,
    "column": 4,
    "beam": 5,
    "objects": 6,
    "misc": 7,
    "unlabeled": 8,
}

INVALID_ID = open3d.t.geometry.RaycastingScene.INVALID_ID  # 4294967295
INVALID_CATEGORY = 41  # the unlabeled index

CATEGORY_TO_HEX = {
    0: "#ffffff",
    1: "#aec7e8",
    2: "#708090",
    3: "#98df8a",
    4: "#c5b0d5",
    5: "#ff7f0e",
    6: "#d62728",
    7: "#1f77b4",
    8: "#bcbd22",
    9: "#ff9896",
    10: "#2ca02c",
    11: "#e377c2",
    12: "#de9ed6",
    13: "#9467bd",
    14: "#8ca252",
    15: "#843c39",
    16: "#9edae5",
    17: "#9c9ede",
    18: "#e7969c",
    19: "#637939",
    20: "#8c564b",
    21: "#dbdb8d",
    22: "#d6616b",
    23: "#cedb9c",
    24: "#e7ba52",
    25: "#393b79",
    26: "#a55194",
    27: "#ad494a",
    28: "#b5cf6b",
    29: "#5254a3",
    30: "#bd9e39",
    31: "#c49c94",
    32: "#f7b6d2",
    33: "#6b6ecf",
    34: "#ffbb78",
    35: "#c7c7c7",
    36: "#8c6d31",
    37: "#e7cb94",
    38: "#ce6dbd",
    39: "#17becf",
    40: "#7f7f7f",
    41: "#000000",
}


def hex_to_rgb(hex_string):
    """Convert a hex color string to an RGB color.

    Parameters
    ----------
    hex_string (str)
        The hex color string.

    Returns
    -------
    numpy array length 3
        The RGB color.
    """
    hex_string = hex_string.lstrip("#")
    return np.array(
        [int(hex_string[i : i + 2], base=16) for i in (0, 2, 4)], dtype=np.uint8
    )


def visualise_segmentation(segmentation_map, save_path):
    image = np.zeros(
        (segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8
    )
    for i in range(segmentation_map.shape[0]):
        for j in range(segmentation_map.shape[1]):
            image[i, j, :] = hex_to_rgb(
                CATEGORY_TO_HEX[segmentation_map[i, j] % len(CATEGORY_TO_HEX)]
            )

    # save image
    img = Image.fromarray(image)
    img.save(save_path)


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


def calc_pixel_triangle_mapping(raycaster, camera_pose, intrinsic_matrix, opts):
    # the camera pose is the camera-to-world transformation, so invert_se3(camera_pose) is the world-to-camera transformation.
    extrinsic_matrix = invert_se3(camera_pose)
    extrinsic_matrix = open3d.core.Tensor(extrinsic_matrix)

    intrinsic_matrix = open3d.core.Tensor(intrinsic_matrix)

    rays = open3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix, extrinsic_matrix, opts.img_width, opts.img_height
    )

    return raycaster.cast_rays(rays)["primitive_ids"].numpy()


def load_triangle_to_cat(opts, scene_dir, triangles, raw_semantic_label_to_mpcat40):
    """Creates maps from triangle indices to segment indices and segment indices to category indices.

    Returns:
    --------
    triangle_to_segment ((n_triangles, ) numpy array):
        the segment index for each triangle

    segment_to_category ((n_segments, ) numpy array):
        the category index for each segment
    """
    scene_name = os.path.basename(scene_dir)

    #### load triangle -> segment mappings
    # this should be a numpy array of shape (num_triangles,), where each entry is the segment index of the triangle
    # NOTE I am just using simple majority voting to turn the vertex classifications into triangle classifications. This is crude but should be fast and good enough for now.

    # load vertex -> segment mappings
    with open(
        os.path.join(scene_dir, f"{scene_name}_vh_clean_2.0.010000.segs.json")
    ) as f:
        vertex_to_segment = np.array(json.load(f)["segIndices"], dtype=np.int32)

    # turn into triangle -> segment mappings
    triangle_corners_to_segment = vertex_to_segment[triangles]
    triangle_to_segment = np.empty(triangle_corners_to_segment.shape[0], dtype=np.int32)

    # first corner is the same as at least one other corner, so in majority
    use_first_corner = (
        triangle_corners_to_segment[:, 0] == triangle_corners_to_segment[:, 1]
    ) | (triangle_corners_to_segment[:, 0] == triangle_corners_to_segment[:, 2])
    triangle_to_segment[use_first_corner] = triangle_corners_to_segment[
        use_first_corner, 0
    ]
    # either second corner is in the majority, or there is no majority in which case arbitrarily use the second corner
    triangle_to_segment[~use_first_corner] = triangle_corners_to_segment[
        ~use_first_corner, 1
    ]

    # add a new segment for invalid ids -- basically all rays that don't hit a triangle hit this appended phantom triangle with a phantom segment
    # this might end up not being needed if they use the "void" category for all this, but I can't be sure of that
    triangle_to_segment = np.concatenate(
        (triangle_to_segment, [triangle_to_segment.max() + 1])
    )

    n_segments = triangle_to_segment.max() + 1

    # a bunch of segments have no object (don't appear in the segGroups file), so start out with putting them all as INVALID_CATEGORY
    if opts.instance_segmentation:
        segment_to_category = np.ones(n_segments, dtype=np.int32) * (-1)
    else:
        segment_to_category = np.ones(n_segments, dtype=np.int32) * INVALID_CATEGORY

    #### load segment -> category mappings
    with open(os.path.join(scene_dir, f"{scene_name}_vh_clean.aggregation.json")) as f:
        seggroups = json.load(f)["segGroups"]

    n_stuff_objects = 0
    for i in range(len(seggroups)):
        if opts.instance_segmentation:
            # different categories for individual instances of objects of the same category
            # add the max category index to the object id to get a unique category index for each instance even across regions
            if seggroups[i]["label"] not in raw_semantic_label_to_mpcat40:
                print(
                    f"Warning: label {seggroups[i]['label']} not found in category mapping, setting category to 15 (unlabeled)."
                )
                category_idx = 15
                n_stuff_objects += 1
            elif (
                raw_semantic_label_to_mpcat40[seggroups[i]["label"]]
                in STUFF_CLASSES.keys()
            ):  # stuff objects are all mapped to their semantic class
                category_idx = STUFF_CLASSES[
                    raw_semantic_label_to_mpcat40[seggroups[i]["label"]]
                ]
                n_stuff_objects += 1
            else:  # not stuff, so every object is its own category
                category_idx = (
                    seggroups[i]["objectId"] + len(STUFF_CLASSES) - n_stuff_objects
                )
        else:
            category_idx = MPCAT40_TO_INDEX[
                raw_semantic_label_to_mpcat40[seggroups[i]["label"]]
            ]

        segment_to_category[seggroups[i]["segments"]] = category_idx

    if opts.instance_segmentation:
        # change the category for invalid objects (the ones that are mapped to -1) to the void category, ie 0
        print(
            f"Setting {np.sum(segment_to_category < 0)} unmapped segments out of {segment_to_category.shape[0]} in total ({np.sum(segment_to_category < 0) / segment_to_category.shape[0] * 100:.3f}%) to 0 (void)."
        )
        segment_to_category[segment_to_category < 0] = 0

    # check that segment_to_category has length num_segments (including the phantom segment, which is already included in triangle_to_segment)
    assert segment_to_category.shape[0] == np.max(triangle_to_segment) + 1
    return segment_to_category[triangle_to_segment]


def load_mesh(scene_dir):
    scene_name = os.path.basename(scene_dir)

    # load the mesh
    scene_mesh = open3d.io.read_triangle_mesh(
        os.path.join(scene_dir, f"{scene_name}_vh_clean_2.ply")
    )
    return scene_mesh


def main(opts):
    raw_semantic_label_to_mpcat40 = {}

    with open(os.path.join(opts.category_mapping_file), "r") as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")

        # Skip header row
        next(reader)
        for row in reader:
            # Map second column to second-to-last column
            raw_semantic_label_to_mpcat40[row[1]] = row[-2]

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

        print(f"Processing scene {scene}")
        scene_dir = os.path.join(opts.raw_dataset_dir, scene)

        if opts.instance_segmentation:
            if not os.path.exists(os.path.join(scene_dir, "object_maps")):
                os.makedirs(os.path.join(scene_dir, "object_maps"))

        else:
            if not os.path.exists(os.path.join(scene_dir, "semantic_maps")):
                os.makedirs(os.path.join(scene_dir, "semantic_maps"))

        scene_mesh = load_mesh(scene_dir)
        triangle_to_category = load_triangle_to_cat(
            opts,
            scene_dir,
            np.asarray(scene_mesh.triangles),
            raw_semantic_label_to_mpcat40,
        )

        # For some reason open3d has more than one TriangleMesh class
        raycaster_mesh = open3d.t.geometry.TriangleMesh(
            open3d.core.Tensor(np.asarray(scene_mesh.vertices).astype(np.float32)),
            open3d.core.Tensor(np.asarray(scene_mesh.triangles).astype(np.float32)),
        )

        raycaster = open3d.t.geometry.RaycastingScene()
        raycaster.add_triangles(raycaster_mesh)

        # check that triangle_to_segment has length num_triangles + 1 (for the phantom triangle)
        assert (
            triangle_to_category.shape[0]
            == np.asarray(scene_mesh.triangles).shape[0] + 1
        )

        # load camera parameters for depths to fill in bad depths
        intrinsic_matrix = np.loadtxt(
            os.path.join(scene_dir, "intrinsic", "intrinsic_color.txt")
        )
        intrinsic_matrix = intrinsic_matrix[:3, :3]

        # adjust for larger pixels
        intrinsic_matrix[0, 0] *= opts.img_width / 1296
        intrinsic_matrix[1, 1] *= opts.img_height / 968
        intrinsic_matrix[0, 2] *= opts.img_width / 1296
        intrinsic_matrix[1, 2] *= opts.img_height / 968

        img_names = set(
            [
                int(name.strip(".txt"))
                for name in sorted(os.listdir(os.path.join(scene_dir, "pose")))
            ]
        )
        max_img_idx = max(img_names)

        # do this for each image, as I need to save each image's category indices separately
        for i in tqdm(
            range(max_img_idx + 1),
            desc="Calculating segmentation maps for scene {}".format(scene),
        ):
            if i not in img_names:
                continue
            camera_pose = np.loadtxt(os.path.join(scene_dir, "pose", f"{i}.txt"))
            if np.isinf(camera_pose).any() or np.isnan(camera_pose).any():
                print(f"Skipping image {i} because the camera pose is invalid.")
                continue
            triangle_indices = calc_pixel_triangle_mapping(
                raycaster, camera_pose, intrinsic_matrix, opts
            )
            invalid_triangles = triangle_indices == INVALID_ID
            n_invalid_triangles = np.sum(invalid_triangles)
            if n_invalid_triangles / (opts.img_height * opts.img_width) > 0.2:
                print(
                    f"Skipping image {i} because {n_invalid_triangles} / {opts.img_height * opts.img_width} ({100 * n_invalid_triangles / (opts.img_height * opts.img_width):.3f}%) rays hit no triangle."
                )
                continue
            triangle_indices[invalid_triangles] = 0  # avoid out of bounds error
            category_indices = triangle_to_category[triangle_indices]
            category_indices[invalid_triangles] = 0  # map invalid triangles void

            if opts.debug:
                if not os.path.exists("./test_segm/{}".format(scene)):
                    os.makedirs("./test_segm/{}".format(scene))
                visualise_segmentation(category_indices, f"./test_segm/{scene}/{i}.png")

                command = input(
                    'Press "Enter" to continue to next image, "ns" to go to the next scene...'
                )
                if command == "b":
                    breakpoint()
                elif command == "ns":
                    break
            else:
                if opts.instance_segmentation:
                    np.save(
                        os.path.join(scene_dir, "object_maps", f"{i}"),
                        category_indices,
                    )
                else:
                    np.save(
                        os.path.join(scene_dir, "semantic_maps", f"{i}"),
                        category_indices,
                    )

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--debug", action="store_true", help="debug mode", default=False
    )
    parser.add_argument(
        "-i",
        "--instance_segmentation",
        action="store_true",
        help="segment individual instances of objects, not categories",
        default=True,
    )
    parser.add_argument(
        "--raw_dataset_dir",
        type=str,
        default="data/scannet/scans",
    )
    parser.add_argument(
        "--category_mapping_file",
        type=str,
        default="data/scannet/scannetv2-labels.combined.tsv",
    )
    parser.add_argument("--img_width", type=int, default=320)
    parser.add_argument("--img_height", type=int, default=240)
    parser.add_argument("--start_scene", type=str, default=None)
    parser.add_argument("--end_scene", type=str, default=None)
    opts = parser.parse_args()
    main(opts)
