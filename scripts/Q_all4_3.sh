#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o ./logs/Q_all4_3.out
#SBATCH --job-name=Q_all4_3

python3 main.py --problem all4 -b 1 -pe 10 -a GELU -res True -v False -net FFJORDNet -ls 100 -nl 5 -l KL_model_control --rescale True -e 16000 -d True -n 480 -w 0.1 -pe 40
