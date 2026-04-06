#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o ./logs/Q_1_well_same_hyperparams_4.out
#SBATCH --job-name=Q_1_well_same_hyperparams_4

#python3 main.py --problem cat     -b 1 -pe 10 -a GELU -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -w 0.1 -d True -n 1000 -e 1000 -cv 1
#python3 main.py --problem 10      -b 1 -pe 10 -a GELU -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -w 0.1 -d True -n 1000 -e 1000 -cv 1
#python3 main.py --problem num_0   -b 1 -pe 10 -a GELU -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -w 0.1 -d True -n 1000 -e 1000 -cv 1
#python3 main.py --problem binom_0 -b 1 -pe 10 -a GELU -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -w 0.1 -d True -n 1000 -e 1000 -cv 1
python3 main.py --problem GKP     -b 1 -pe 10 -a GELU -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -w 0.1 -d True -n 5000 -e 1000 -cv 1