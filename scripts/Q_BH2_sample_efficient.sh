#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o ./logs/Q_BH2_sample_efficient.out
#SBATCH --job-name=Q_BH2_sample_efficient

python3 main.py --problem BH2 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -e 1000 -d True -n 10000 -spr 1000