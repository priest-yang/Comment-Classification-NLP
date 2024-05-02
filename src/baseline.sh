#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=test1
#SBATCH --account=si630w24_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16000m
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=demo.out

# The application(s) to execute along with its input arguments and options:

# /bin/hostname
# echo “hello world”
# nvidia-smi
python3 baseline.py