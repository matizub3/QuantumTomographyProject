#!/bin/bash

#SBATCH --job-name 10_W_3_2
#SBATCH -o ./logs/10_W_3_2.out
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem 200G
#SBATCH --time=2-00:00:00

pixi run python -u main.py \
  -r W \
  --problem 10_W \
  -b 1 \
  -pe 500 \
  -a GELU \
  -res True \
  --rescale True \
  --rescale_mult 1.0 \
  -v False \
  -net FFJORDNet \
  -ls 150 \
  -nl 5 \
  -l L1_efficient \
  -e 25000 \
  -lr 0.001 \
  -d True \
  -w 0.1 \
  -spr 100 \
  -re 10
