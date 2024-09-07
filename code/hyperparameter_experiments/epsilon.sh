#!/bin/bash

# randomly selected seeds
seeds=(923936036 1841009588 1579824767 862396893 1126136760 2053152965 1650170623 460952620 855928699 29318224)

for seed in "${seeds[@]}"; do
  for eps in 0.0001 0.001 0.01 0.1; do
    python certified_training.py\
      --batch-size 100\
      --learning-rate 0.5\
      --epochs 200\
      --pretrain-epochs 0\
      --epsilon "$eps"\
      --hidden-nodes 20\
      --dataset moons\
      --experiment 2moons_eps_"$eps"_seed_"$seed"_test\
      --seed "$seed"\
      --loss bce\
      --perturb-inference
  done
done
