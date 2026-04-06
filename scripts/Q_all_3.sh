#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o ./logs/Q_all_3.out
#SBATCH --job-name=Q_all_3

python3 main.py --problem all -b 1 -pe 40 -a GELU -res True -v False -net FFJORDNet -ls 80 -nl 5 -l KL_efficient --rescale True -e 2000 -d True -n 1000 -spr 1000 -w 0.4 -lr 0.001 -cv 1