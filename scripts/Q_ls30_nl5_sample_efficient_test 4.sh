#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o ./logs/Q_ls30_nl5_sample_efficient_test2.out
#SBATCH --job-name=Q_ls30_nl5_sample_efficient_test2

python3 main.py --problem cat -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -d True -n 1000
python3 main.py --problem 10 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True  -d True -n 1000
python3 main.py --problem num_0 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -d True -n 1000
python3 main.py --problem binom_0 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -d True -n 1000
python3 main.py --problem GKP -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -e 1000 -d True -n 1000