#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH -o ./logs/Q_multi_num_14.out
#SBATCH --job-name=Q_multi_num_14

#python3 main.py --problem Multi_Num -nw 2 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -e 1000 -d True -n 2000 -spr 100 -w 0.2
#python3 main.py --problem Multi_Num -nw 3 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -e 1000 -d True -n 2000 -spr 100 -w 0.2
#python3 main.py --problem Multi_Num -nw 6 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 80 -nl 5 -l KL_efficient --rescale True -e 1000 -d True -n 2000 -spr 1000 -w 0.4
python3 main.py --problem Multi_Num -nw 10 -b 1 -pe 10 -a GELU -res True -v False -net FFJORDNet -ls 80 -nl 5 -l KL_model_control --rescale True -e 4000 -d True -n 500 -w 0.1 -pe 40
#python3 main.py --problem Multi_Num -nw 5 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -e 1000 -d True -n 2000 -spr 100 -w 0.2
#python3 main.py --problem Multi_Num -nw 6 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -e 2000 -d True -n 2000 -spr 1000 -w 0.4
#python3 main.py --problem Multi_Num -nw 7 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -e 1000 -d True -n 2000 -spr 100 -w 0.2
#python3 main.py --problem Multi_Num -nw 8 -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -e 1000 -d True -n 2000 -spr 1000 -w 0.2