#!/bin/bash
#SBATCH --job-name=edm_classifier
#SBATCH --partition=gpu-l40
#SBATCH --account=stf
#SBATCH --mail-user=songk42@uw.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/edm_classifier_%j.log
#SBATCH --time=3-00:00:00

module load cuda/12.9.1
module load conda/Miniforge3-25.9.1-0

cd /mmfs1/home/songk42/e3_diffusion_for_molecules
conda activate edm

cd qm9/property_prediction

WANDB_MODE=offline python main_qm9_prop.py --num_workers 2 --lr 5e-4 --property relative_atomic_energy --exp_name exp_class_relenergy --model_name egnn --checkpoint_interval 10  --resume latest
