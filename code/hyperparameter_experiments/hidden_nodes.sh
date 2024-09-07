#!/bin/bash

python certified_training.py\
  --batch-size 100\
  --learning-rate 0.5\
  --epochs 200\
  --pretrain-epochs 0\
  --epsilon 0.01\
  --hidden-nodes 40\
  --dataset moons\
  --perturbation all\
  --experiment 2moons_2_layer_20_nodes_test\
  --seed 10123310\
  --loss bce\
  --perturb-inference
