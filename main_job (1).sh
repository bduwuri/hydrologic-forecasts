#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=20        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:a100:1           # number of gpus per node
#SBATCH --job-name=ttf         # create a short name for your job
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mem=50G

#module purge


#conda create --name optuna_ttf python=3.9 -y
source ~/.bashrc
conda init bash

conda activate tf212-gpu
module load cuda/11.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/duvvuri.b/.local/lib/python3.9/site-packages/tensorrt
export LD_LIBRARY_PATH=/home/duvvuri.b/.conda/envs/tf212-gpu/lib:$LD_LIBRARY_PATH

#pip install torch==2.0.1 pytorch-lightning==2.0.2 pytorch_forecasting==1.0.0 torchaudio==2.0.2 torchdata==0.6.1 torchtext==0.15.2 torchvision==0.15.2 optuna==3.4
#conda install pytorch-cuda=11.7 -c nvidia -y

# cp -R tuning.py /home/duvvuri.b/.local/lib/python3.9/site-packages/pytorch_forecasting/models/temporal_fusion_transformer/tuning.py
# debugging flags (optional)
##export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest cuda
#module load nccl/2.8.3-1-cuda.11.8

export TF_CPP_MIN_LOG_LEVEL=2

python script_hyperparam_opt.py