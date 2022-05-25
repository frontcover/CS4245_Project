!/bin/sh
#
#SBATCH --job-name="Move_data"
#SBATCH --partition=trans
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

rsync -a /tudelft.net/staff-groups/ewi/me/MS3/Studenten/'Simin Zhu'/SLAM/Dataset /scratch/szhu2/

