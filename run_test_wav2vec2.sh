#!/bin/bash

#SBATCH --job-name=test_asr_wav2vec2
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-type=START,END,FAIL
#SBATCH --account=education-eemcs-courses-dsait4095

# Load modules:
module load 2023r1
module load cuda/12.5
module load openmpi
module load py-pip
module load py-numpy
module load py-pyyaml
module load py-tqdm
module load miniconda3
module load ffmpeg
conda activate /home/kmjones/.conda/envs/example

python train_with_wav2vec.py hparams/train_wav2vec2.yaml --test_only

conda deactivate