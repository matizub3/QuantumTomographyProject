#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o ./logs/W_1_well_same_hyperparams_2.out
#SBATCH --job-name=W_1_well_same_hyperparams_2

python3 main.py -r W --problem binom_0 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l L1_efficient --rescale True -w 0.05 -d True -n 1000 -e 1000
python3 main.py -r W --problem binom_0 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l L1_efficient --rescale True -w 0.2 -d True -n 1000 -e 1000
#python3 main.py -r W --problem GKP     -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l L1_efficient --rescale True -w 0.1 -d True -n 1000 -e 4000 
#python3 main.py -r W --problem 10      -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l L1_efficient --rescale True -w 0.1 -d True -n 1000 -e 4000 