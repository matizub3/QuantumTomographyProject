#!/bin/bash

#SBATCH --job-name Q_multi_num
#SBATCH -o ./logs/Q_multi_num.out
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p lulab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem 200G
#SBATCH --time=2-00:00:00

cd /cluster/tufts/lulab/mzubrz01/QuantumTomographyProject || exit 1

pixi run python main.py --problem Multi_Num -nw 10 -b 1 -pe 10 -a GELU -res True -v False -net FFJORDNet -ls 80 -nl 5 -l KL_model_control --rescale True -e 8000 -d True -n 500 -w 0.1 -pe 40
