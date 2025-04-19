#!/bin/bash

dataset_dir="data/matterport3d"

echo "Running Matterport3D pipeline"

echo "Preprocessing Depth and Images"

python scripts/data_preparation_scripts/matterport3d_precompute.py --raw_dataset_dir $dataset_dir --img_width 320 --img_height 256 --color_dir_name rgb --depth_dir_name depth

echo "Finished preprocessing Depth and Images"

echo "Now computing the 3D patch-locations"

python scripts/data_preparation_scripts/matterport3d_calc_3dpatchlocations.py --raw_dataset_dir $dataset_dir --img_width 320 --img_height 256 --patch_size 8 

echo "Finished computing the 3D patch-locations"

echo "Computing image overlaps..."

python scripts/data_preparation_scripts/matterport3d_calc_img_overlap.py --raw_dataset_dir $dataset_dir --img_width 40 --img_height 32 --patch_location_file patch_locations_32_40.npy

echo "Finished computing image overlaps"

echo "Now computing object segmentation maps"

python scripts/data_preparation_scripts/matterport3d_compute_segmentations.py --raw_dataset_dir $dataset_dir --img_width 320 --img_height 256 --instance_segmentation 

echo "Finished computing object segmentation maps"