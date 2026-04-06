#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o ./logs/Q_all3.out
#SBATCH --job-name=Q_all3

python3 main.py --problem all3 -b 1 -pe 10 -a GELU -res True -v False -net FFJORDNet -ls 80 -nl 5 -l KL_model_control --rescale True -e 32000 -d True -n 600 -w 0.1 -pe 40 -lr 0.001
