#!/bin/bash

for lr in 0.01 0.1 1.0 5.0 10.0; do
  python certified_training.py\
    --batch-size 100\
    --learning-rate "$lr"\
    --epochs 500\
    --pretrain-epochs 0\
    --epsilon 0.01\
    --hidden-nodes 20\
    --dataset moons\
    --perturbation all\
    --experiment 2moons_lr_"$lr"_test\
    --seed 10123310\
    --loss bce\
    --perturb-inference
done
