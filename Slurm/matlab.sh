#!/bin/sh
#
#SBATCH --job-name="matlab_preprocessing"
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

module load 2022r1
module load matlab

srun matlab -batch "run('/home/szhu2/CS4245_Project/MATLAB/main.m'); exit;"
