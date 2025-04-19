#!/bin/bash
dataset_dir="data/scannet"

echo "Running ScanNet pipeline"
echo "Extracting Images and Depths from .sens files for training scenes..."

python $dataset_dir/reader.py --filepath $dataset_dir/scans --output_path $dataset_dir/scans --export_depth_images --export_color_images --export_poses --export_intrinsics

echo "Extracting Images and Depths from .sens files for testing scenes..."

python $dataset_dir/reader.py --filepath $dataset_dir/scans_test --output_path $dataset_dir/scans_test --export_depth_images --export_color_images --export_poses --export_intrinsics

echo "Finished extracting Images and Depths from .sens files"

echo "Preprocessing Depth and Images for training scenes..."

python scripts/data_preparation_scripts/scannet_precompute.py --raw_dataset_dir /scratch/shared/beegfs/dominik/scannet/scans --img_width 320 --img_height 240 --color_dir_name color_npy --depth_dir_name depth_npy

echo "Preprocessing Depth and Images for test scenes..."

python scripts/data_preparation_scripts/scannet_precompute.py --raw_dataset_dir /scratch/shared/beegfs/dominik/scannet/scans_test --img_width 320 --img_height 240 --color_dir_name color_npy --depth_dir_name depth_npy

echo "Finished preprocessing Depth and Images"

echo "Now computing the 3D patch-locations for training scenes..."

python scripts/data_preparation_scripts/scannet_calc_3dpatchlocations.py --raw_dataset_dir /scratch/shared/beegfs/dominik/scannet/scans --img_width 320 --img_height 240 --patch_size 8

echo "Now computing the 3D patch-locations for test scenes..."

python scripts/data_preparation_scripts/scannet_calc_3dpatchlocations.py --raw_dataset_dir /scratch/shared/beegfs/dominik/scannet/scans_test --img_width 320 --img_height 240 --patch_size 8

echo "Computing image overlaps for training scenes..."

python scripts/data_preparation_scripts/scannet_calc_img_overlap.py --raw_dataset_dir /scratch/shared/beegfs/dominik/scannet/scans --img_width 40 --img_height 30 --patch_location_file patch_locations_30_40.npy

echo "Computing image overlaps for test scenes..."

python scripts/data_preparation_scripts/scannet_calc_img_overlap.py --raw_dataset_dir /scratch/shared/beegfs/dominik/scannet/scans_test --img_width 40 --img_height 30 --patch_location_file patch_locations_30_40.npy

echo "Finished computing image overlaps"

echo "Now computing object segmentation maps for training scenes..."

python scripts/data_preparation_scripts/scannet_compute_segmentations.py --raw_dataset_dir /scratch/shared/beegfs/dominik/scannet/scans --img_width 320 --img_height 240 --instance_segmentation

echo "Finished computing object segmentation maps for training scenes"
