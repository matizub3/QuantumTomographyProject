#!/bin/bash

#SBATCH --job-name W_experimental
#SBATCH -o ./logs/W_experimental.out
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p lulab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem 200G
#SBATCH --time=1-00:00:00

cd /cluster/tufts/lulab/mzubrz01/QuantumTomographyProject || exit 1

pixi run python main.py --problem QST_CGAN_W_Neg -r W -b 1 -pe 10 -a GELU -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_model_control --rescale True -e 1000 -d True -n 1000 -w 0.1 -pe 10