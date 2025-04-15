#!/bin/bash

#SBATCH --job-name=setup
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-eemcs-courses-dsait4095

# Load modules:
module load miniconda3
conda activate /scratch/mfron/IST-LAB/IST-ASR-3

module load 2023r1
module load cuda/12.5
module load openmpi
module load py-torch/1.12.1
module load py-pip
module load py-numpy
module load py-pyyaml
module load py-tqdm
module load ffmpeg

srun python train_tokenizer.py hparams/tokenizer.yaml

conda deactivate