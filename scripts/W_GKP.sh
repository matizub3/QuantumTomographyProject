#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o ./logs/W_GKP.out
#SBATCH --job-name=W_GKP

python3 main.py --problem GKP -r W -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l L1_model_average --rescale True -d True -n 5000