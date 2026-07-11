#!/bin/bash
#SBATCH --job-name=DeepSAVAtest
#SBATCH --error=deepsava.%j.err
#SBATCH --output=deepsava.%j.out
#SBATCH --partition=debug
#SBATCH --account=thesis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=mps:30
#SBATCH --time=00:30:00

module load singularity

PROJECT_DIR=/home/arbevila/projects/DeepSAVA
CONTAINER=${PROJECT_DIR}/cluster/deepsava.sif
VIDEO_BIND=/home/arbevila/storage
VIDEO=/storage/datasets/ucf101/test/Basketball/v_BasketballDunk_g04_c03.avi
OUTPUT=${PROJECT_DIR}/output/I3D_test

cd ${PROJECT_DIR}
mkdir -p ${OUTPUT}

singularity exec --nv --bind ${PROJECT_DIR}:/workspace,${VIDEO_BIND}:/storage ${CONTAINER} \
    python /workspace/test_gen.py \
      -o /workspace/output/I3D_test \
      --model i3d_inception \
      --dataset UCF101 \
      --video_path ${VIDEO} \
      --video_class Basketball \
      --constraint iteration \
      --num_iter 1 \
      --save_freq 1
