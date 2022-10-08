#!/bin/bash

#SBATCH --job-name="rxj1131"
#SBATCH --output=/home/ajshajib/Logs"/joblog.%j"
#SBATCH --error=/home/ajshajib/Logs"/error.%j"
#SBATCH --partition=broadwl
#SBATCH --account=pi-jfrieman
#SBATCH -t 36:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=28
#SBATCH --mem-per-cpu=2000
#SBATCH --exclusive
#SBATCH --mail-user=ajshajib@uchicago.edu
#SBATCH --mail-type=ALL

$software="$1"
$ani_model="$2"
$slit="$3"
$spherical="$4"
$lens_model="$5"

module load gcc/7.2.0
source /home/ajshajib/.bashrc

which conda
which python3

mpirun -np 56 python3 /home/ajshajib/RXJ1131_kinematics/run_mcmc.py $software $ani_model $slit $spherical $lens_model >& /home/ajshajib/Logs/output.$SLURM_JOB_ID