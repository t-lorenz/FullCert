#!/bin/bash

for bs in 50 100 500 1000; do
  python certified_training.py\
    --batch-size "$bs"\
    --learning-rate 0.5\
    --epochs 1000\
    --pretrain-epochs 0\
    --epsilon 0.01\
    --hidden-nodes 20\
    --dataset moons\
    --perturbation all\
    --experiment 2moons_bs_"$bs"_test\
    --seed 10123310\
    --loss bce\
    --perturb-inference
done
