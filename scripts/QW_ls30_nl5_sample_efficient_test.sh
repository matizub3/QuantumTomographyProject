#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o ./logs/QW_ls30_nl5_sample_efficient_test.out
#SBATCH --job-name=QW_ls30_nl5_sample_efficient_test

python3 main.py --problem 10 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -n 1000
python3 main.py --problem binom_0 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -n 1000
python3 main.py --problem num_0 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -n 1000

python3 main.py --problem BH2 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -n 1000
python3 main.py --problem W -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -n 1000