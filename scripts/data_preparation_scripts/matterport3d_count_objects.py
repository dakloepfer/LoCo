"""Count objects and stuff class objects in Matterport3D dataset for each scene."""

import argparse
import csv
import json
import os

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


def count_objects_one_scene(all_regions, scene_dir, raw_semantic_label_to_mpcat40):
    """Counts the number of valid objects in a single scene, and the number of stuff objects per stuff object class.

    Parameters
    ----------
    all_regions (list of str):
        the names of the regions that make up the scene

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
    n_regions = len(all_regions)

    all_seggroups = []
    for region in all_regions:
        with open(
            os.path.join(scene_dir, "region_segmentations", region + ".semseg.json")
        ) as f:
            all_seggroups.append(json.load(f)["segGroups"])

    total_stuff_objects = 0
    total_all_objects = 0
    total_valid_objects = 0
    stuff_objects_per_class = [0] * len(STUFF_CLASSES)

    for i in range(n_regions):

        total_all_objects += len(all_seggroups[i])
        for j in range(len(all_seggroups[i])):
            # different categories for individual instances of objects of the same category
            # add the max category index to the object id to get a unique category index for each instance even across regions
            if all_seggroups[i][j]["label"] not in raw_semantic_label_to_mpcat40:
                # count as unlabelled, ie "stuff"
                stuff_objects_per_class[STUFF_CLASSES["unlabeled"]] += 1
                total_stuff_objects += 1
            elif (
                raw_semantic_label_to_mpcat40[all_seggroups[i][j]["label"]]
                in STUFF_CLASSES.keys()
            ):
                stuff_objects_per_class[
                    STUFF_CLASSES[
                        raw_semantic_label_to_mpcat40[all_seggroups[i][j]["label"]]
                    ]
                ] += 1
                total_stuff_objects += 1

            else:
                total_valid_objects += 1

    assert total_valid_objects + total_stuff_objects == total_all_objects
    assert sum(stuff_objects_per_class) == total_stuff_objects

    return total_valid_objects, stuff_objects_per_class


def main(opts):
    raw_semantic_label_to_mpcat40 = {}

    with open(
        os.path.join(opts.category_mapping_dir, "category_mapping.tsv"), "r"
    ) as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")

        # Skip header row
        next(reader)
        for row in reader:
            # Map second column to last column
            raw_semantic_label_to_mpcat40[row[1]] = row[-1]

    n_valid_objects_per_scene = {}
    n_stuff_objects_per_scene = {}

    for scene in tqdm(ALL_SCENES, "Counting objects in each scene..."):
        scene_dir = os.path.join(opts.raw_dataset_dir, scene)

        all_files = os.listdir(os.path.join(scene_dir, "region_segmentations"))
        all_regions = list(
            set([f.split(".")[0] for f in all_files if f.startswith("region")])
        )
        all_regions.sort()

        n_valid_objects, n_stuff_objects = count_objects_one_scene(
            all_regions, scene_dir, raw_semantic_label_to_mpcat40
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
        default="data/matterport3d",
    )
    parser.add_argument(
        "--category_mapping_dir",
        type=str,
        help="Path to the directory containing the category mapping file",
        default="./scripts/data_preparation_scripts",
    )

    opts = parser.parse_args()
    main(opts)
