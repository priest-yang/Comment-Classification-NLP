#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=batchsize1
#SBATCH --account=si630w24_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16000m
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=demo.out

# The application(s) to execute along with its input arguments and options:

# /bin/hostname
# echo “hello world”
# nvidia-smi
module load python/3.11.5
python3 train_HierAtt.py