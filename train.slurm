#!/bin/bash
#SBATCH --job-name=edm_train
#SBATCH --partition=gpu-l40s
#SBATCH --account=stf
#SBATCH --mail-user=songk42@uw.edu
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/edm_train_%j.log
#SBATCH --time=3-00:00:00

module load cuda/12.9.1
module load conda/Miniforge3-25.9.1-0

cd /mmfs1/home/songk42/e3_diffusion_for_molecules
conda activate edm

#cd qm9/property_prediction

python main_qm9.py --exp_name exp_cond_gap  --model egnn_dynamics --lr 1e-4  --nf 192 --n_layers 9 --save_model True --diffusion_steps 1000 --sin_embedding False --n_epochs 3000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 64 --normalize_factors [1,8,1] --conditioning gap --dataset qm9_second_half --online False --resume latest --no_wandb
