#!/bin/bash

#SBATCH --job-name Q_1_well
#SBATCH -o ./logs/Q_1_well.out
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem 200G
#SBATCH --time=1-00:00:00

python3 main.py --problem cat     -b 1 -pe 10 -a GELU -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -w 0.1 -d True
python3 main.py --problem 10      -b 1 -pe 10 -a GELU -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -w 0.1 -d True
python3 main.py --problem num_0   -b 1 -pe 10 -a GELU -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -w 0.1 -d True
python3 main.py --problem binom_0 -b 1 -pe 10 -a GELU -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -w 0.1 -d True
python3 main.py --problem GKP     -b 1 -pe 10 -a GELU -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -w 0.1 -d True -e 1000 