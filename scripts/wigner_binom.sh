#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o ./logs/wigner_binom.out
#SBATCH --job-name=wigner_binom

python3 main.py --problem 10 -b 1 -pe 10 -a SINE -res True -n 10000 -v False -net FFJORDNet -nl 5 -l L1_model_average -r W --rescale True -e 1000 -d True