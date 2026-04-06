#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o ./logs/W_ls30_nl5_sample_efficient_test3.out
#SBATCH --job-name=W_ls30_nl5_sample_efficient_test3

python3 main.py -r W --problem binom_0 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l L1_efficient -e 1000 -lr 0.001 -d True -w 0.2 --rescale True -n 1000
python3 main.py -r W --problem 10 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l L1_efficient -e 4000 -lr 0.001 -d True -w 0.2 --rescale True -n 1000
python3 main.py -r W --problem GKP -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l L1_efficient -e 4000 -lr 0.001 -d True -w 0.2 --rescale True -n 1000