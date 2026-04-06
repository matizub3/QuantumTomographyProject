#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o ./logs/wigner_10_n5000_ls30_e4000.out
#SBATCH --job-name=wigner_10_n5000_ls30_e4000

python3 main.py --problem 10 -b 1 -pe 10 -a SINE -res True -n 5000 -v False -net FFJORDNet -nl 5 -ls 30 -l L1_model_average -r W --rescale True -e 4000 -d True