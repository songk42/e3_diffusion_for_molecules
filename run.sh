#!/bin/bash

counter=2
for cutoff in 1.0 1.5 2.0 3.0 5.0 10.0
do
    counter=$((counter+1))
    CUDA_VISIBLE_DEVICES=$counter python main_qm9.py --nn_cutoff=$cutoff  --n_epochs 300 --exp_name "edm_qm9/nn_cutoff=$cutoff" --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 10 --ema_decay 0.9999 &
done