#!/bin/bash
#SBATCH --partition=SCSEGPU_MSAI
#SBATCH --qos=q_msai2x24
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=11G
#SBATCH --job-name=naic
#SBATCH --output=naic.out
#SBATCH --error=naic.err

module load anaconda
source activate advent
python main.py
