#!/bin/bash
#SBATCH --mail-user=salari.m1375@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mori_gpu
#SBATCH --job-name=Orthogonal
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --time=00:59:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:v100l:4

cd $SLURM_TMPDIR
cp -r ~/scratch/Orthogonal .
cd Orthogonal

module load python/3.7 cuda/10.0
virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

# python main.py
python validate_null_space.py


# cp null_space.npy ~/scratch/Orthogonal
# cp model.pt ~/scratch/Orthogonal
