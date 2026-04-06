#!/bin/bash

#SBATCH --job-name W_synthetic
#SBATCH -o ./logs/W_synthetic.out
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p lulab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem 200G
#SBATCH --time=2-00:00:00

cd /cluster/tufts/lulab/mzubrz01/QuantumTomographyProject || exit 1

pixi run python main.py -r W --problem cat     -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l L1_efficient --rescale True -w 0.1 -d True
pixi run python main.py -r W --problem num_0   -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l L1_efficient --rescale True -w 0.1 -d True
pixi run python main.py -r W --problem binom_0 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l L1_efficient --rescale True -w 0.1 -d True -e 1000
pixi run python main.py -r W --problem GKP     -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l L1_efficient --rescale True -w 0.1 -d True -e 4000 
pixi run python main.py -r W --problem 10      -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l L1_efficient --rescale True -w 0.1 -d True -e 4000 