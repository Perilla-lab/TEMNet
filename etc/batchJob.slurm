#!/bin/bash
#SBATCH --partition=GPU-AI
#SBATCH --gres=gpu:volta16:2
#SBATCH --nodes=1
#SBATCH --time=2:00:00

cd /home/hbeatson/
module load singularity
singularity exec runtimeEnv.simg python /home/hbeatson/jupyter_runtime_dir/HIV/HIV_scan_classifier.py -p 2 -lr 0.00015 -b 5 -e 50
pwd
singularity exec runtimeEnv.simg python /home/hbeatson/jupyter_runtime_dir/HIV/HIV_scan_classifier.py -p 2 -lr 0.00015 -b 15 
