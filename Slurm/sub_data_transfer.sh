#!/bin/sh
#
#SBATCH --job-name="js_data"
#SBATCH --partition=trans
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

rsync -av /tudelft.net/staff-groups/ewi/me/MS3/Studenten/'Simin Zhu'/SLAM/Dataset /scratch/szhu2/ 
