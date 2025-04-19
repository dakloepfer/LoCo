#!/bin/bash -l
#SBATCH --job-name=mp_colmap      # Job name
#SBATCH --mail-type=BEGIN,END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=dominik@robots.ox.ac.uk    # Where to send mail
#SBATCH --nodes=1                           # Node count
#SBATCH --ntasks-per-node=1                          # Total number of tasks across all nodes
#SBATCH --cpus-per-task=48               # Number of CPU cores per task
#SBATCH --mem=48gb                          # Job memory request
#SBATCH --time=48:00:00                     # Time limit hrs:min:sec
#SBATCH --partition=low-prio-gpu                     # Partition (compute (default) / gpu)
#SBATCH --gres=gpu:1                       # Requesting 4 GPUs
#SBATCH --output=/users/dominik/locus-dev/exp_traces/matterport_colmap.txt # Console output
#SBATCH --constraint=gmem24G           # Request nodes with specific features 
# ------------------------------

export LD_LIBRARY_PATH=/work/yashsb/apps/cuda-11.8/lib64:/work/eldar/apps/colmap

# Define the base directory
BASE_DIR="/scratch/shared/beegfs/dominik/matterport3d"

# List of scenes
TRAIN_SCENES=("r47D5H71a5s" "sKLMLpTHeUy" "VFuaQ6m2Qom" "sT4fr6TAbpF" "gTV8FGcVJC9" "VVfe2KiqLaN" "XcA2TqTSSAj" "Vvot9Ly1tCj" "E9uDoFAP3SH" "5LpN3gDmAk7" "JF19kD82Mey" "uNb9QFRL6hY" "VLzqgDo317F" "ZMojNkEp431" "s8pcmisQ38h" "1LXtFkjw3qL" "PX4nDJXEHrG" "mJXqzFtmKg4" "SN83YJsR3w2" "kEZ7cmS4wCh" "8WUmhLawc2A" "e9zR4mvMWw7" "qoiz87JEwZ2" "759xd9YjKW5" "7y3sRwLe3Va" "vyrNrziPKCB" "aayBHfsNo7d" "b8cTxDM8gDG" "ur6pFq6Qu1A" "29hnd4uzFmX" "i5noydFURQK" "dhjEzFoUFzH" "D7G3Y4RVNrH" "D7N2EKCX4Sj" "S9hNv5qa7GM" "r1Q1Z4BcV1o" "rPc6DW4iMge" "gZ6f7yhEvPG" "ac26ZMwG7aT" "17DRP5sb8fy" "82sE5b5pLXE" "Pm6F8kyY3z2" "ULsKaCPVFJR" "Uxmj2M2itWa" "JeFG25nYj2p" "V2XKFyX4ASd" "YmJkqBEsHnH" "1pXnuDYAj8r" "EDJbREhghzL" "p5wJjkQkbXX" "pRbA3pwrgk9" "jh4fc5c5qoQ" "VzqfbhrpDEA" "B6ByNegPMKs" "JmbYfDe2QKZ" "2n8kARJN3HM" "PuKPg4mmafe" "cV4RVeZvu5T" "5q7pvUzZiY")

for scene in "${TRAIN_SCENES[@]}"; do
    scene_dir="${BASE_DIR}/${scene}"

    # Create a new directory "colmap_output" as a subfolder of base_dir/scene
    output_dir="${scene_dir}/colmap_output"
    rm -r "$output_dir"
    mkdir -p "$output_dir"
    
    echo "Running colmap for $scene..."
    echo "==========================="

    /work/eldar/apps/colmap/colmap database_creator --database_path ${output_dir}/database.db

    /work/eldar/apps/colmap/colmap feature_extractor --image_path ${scene_dir}/undistorted_color_images --database_path ${output_dir}/database.db --ImageReader.camera_model PINHOLE --SiftExtraction.use_gpu 1

    /work/eldar/apps/colmap/colmap exhaustive_matcher --database_path ${output_dir}/database.db --SiftMatching.max_num_matches 16000 --SiftMatching.use_gpu 0

    mkdir -p ${output_dir}/sparse

    /work/eldar/apps/colmap/colmap mapper --image_path ${scene_dir}/undistorted_color_images --database_path ${output_dir}/database.db --output_path ${output_dir}/sparse
    # should have camera poses from now

    echo "Processing for $scene completed."
done