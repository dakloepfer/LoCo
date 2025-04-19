"""Count objects and stuff class objects in Matterport3D dataset for each scene."""

import argparse
import csv
import json
import os

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
INVALID_CATEGORY = 41  # the unlabeled index


def count_objects_one_scene(scene_dir, raw_semantic_label_to_mpcat40):
    """Counts the number of valid objects in a single scene, and the number of stuff objects per stuff object class.

    Parameters
    ----------
    scene_dir (str):
        the directory to the scene

    raw_semantic_label_to_mpcat40 (dict of str to str):
        maps the raw semantic label to the Matterport40 category

    Returns
    -------
    total_valid_objects (int):
        the total number of valid objects in the scene

    stuff_objects_per_class (list of int):
        the number of stuff objects per stuff object class
    """
    scene_name = os.path.basename(scene_dir)
    with open(os.path.join(scene_dir, f"{scene_name}_vh_clean.aggregation.json")) as f:
        seggroups = json.load(f)["segGroups"]

    total_all_objects = len(seggroups)
    total_stuff_objects = 0
    total_valid_objects = 0
    stuff_objects_per_class = [0] * len(STUFF_CLASSES)

    for seggroup in seggroups:
        # different categories for individual instances of objects of the same category
        # add the max category index to the object id to get a unique category index for each instance even across regions
        if seggroup["label"] not in raw_semantic_label_to_mpcat40:
            # count as unlabelled, ie "stuff"
            stuff_objects_per_class[STUFF_CLASSES["unlabeled"]] += 1
            total_stuff_objects += 1
        elif raw_semantic_label_to_mpcat40[seggroup["label"]] in STUFF_CLASSES.keys():
            stuff_objects_per_class[
                STUFF_CLASSES[raw_semantic_label_to_mpcat40[seggroup["label"]]]
            ] += 1
            total_stuff_objects += 1

        else:
            total_valid_objects += 1

    assert total_valid_objects + total_stuff_objects == total_all_objects
    assert sum(stuff_objects_per_class) == total_stuff_objects

    return total_valid_objects, stuff_objects_per_class


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

    n_valid_objects_per_scene = {}
    n_stuff_objects_per_scene = {}

    for scene in tqdm(all_scenes, "Counting objects in each scene..."):
        scene_dir = os.path.join(opts.raw_dataset_dir, scene)

        n_valid_objects, n_stuff_objects = count_objects_one_scene(
            scene_dir, raw_semantic_label_to_mpcat40
        )

        n_valid_objects_per_scene[scene] = n_valid_objects
        n_stuff_objects_per_scene[scene] = n_stuff_objects

    with open(os.path.join(opts.raw_dataset_dir, "n_valid_objects.json"), "w+") as f:
        json.dump(n_valid_objects_per_scene, f, indent=4)

    with open(os.path.join(opts.raw_dataset_dir, "n_stuff_objects.json"), "w+") as f:
        json.dump(n_stuff_objects_per_scene, f, indent=4)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dataset_dir",
        type=str,
        help="Path to the Matterport3D dataset directory",
        default="data/scannet/scans",
    )
    parser.add_argument(
        "--category_mapping_file",
        type=str,
        help="Path to the directory containing the category mapping file",
        default="data/scannet/scannetv2-labels.combined.tsv",
    )

    opts = parser.parse_args()
    main(opts)
