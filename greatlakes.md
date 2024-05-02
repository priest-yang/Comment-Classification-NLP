# GreakLakes Basic Commands

## connect
``ssh shaozey@greatlakes.arc-ts.umich.edu``

## load python
``module load python/3.11.5``

## file transfer
``scp -r shaozey@greatlakes-xfer.arc-ts.umich.edu:si630/ /Users/shawn/Downloads/Homework\ 3/results``

``$ scp <local file path> <uniqname>@greatlakes-xfer.arc-ts.umich.edu:<destination path on Great Lakes>``

``$ scp -r <local directory path> <uniqname>@greatlakes-xfer.arc-ts.umich.edu:<destination path on Great Lakes>``


## interactive window

``srun --gpus-per-node=1 --job-name=gljob1 --account=si630w24_class --partition=gpu --time=02:00:00 --nodes=1 --ntasks-per-node=1 --mem-per-cpu=16g --partition=gpu --pty /bin/bash -l``


## Submit jobs

``sbatch <shell script name> .sh``

## example .sh

```
#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=test1
#SBATCH --account=si630w24_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=demo.out

# The application(s) to execute along with its input arguments and options:

# /bin/hostname
# echo “hello world”
# nvidia-smi
python3 multi.py
```