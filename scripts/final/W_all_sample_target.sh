#!/bin/bash

#SBATCH --job-name W_all
#SBATCH -o ./logs/W_all.out
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p lulab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem 200G
#SBATCH --time=2-00:00:00

cd /cluster/tufts/lulab/mzubrz01/QuantumTomographyProject || exit 1

#python3 main.py -r W --problem all3 -b 1 -pe 10 -a GELU -res True -v False -net FFJORDNet -ls 50 -nl 5 -l L1_target --rescale True -e 16000 -d True -n 5000 -w 0.1 -pe 40 -lr 5.e-4
pixi run python main.py -r W --problem all3 -b 1 -pe 10 -a GELU -res True -v False -net FFJORDNet -ls 30 -nl 5 -l L1_target --rescale True -e 1000 -d True -n 1000 -w 0.1 -pe 10 -lr 4.e-3
