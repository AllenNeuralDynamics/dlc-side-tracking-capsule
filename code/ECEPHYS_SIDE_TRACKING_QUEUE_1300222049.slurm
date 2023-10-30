#!/bin/bash
#SBATCH --partition=braintv
#SBATCH --qos=production
#SBATCH --nodes=1 --cpus-per-task=4 --gpus=1 --mem=64G
#SBATCH --time=8:00:00
#SBATCH --export=NONE
#SBATCH --job-name=ECEPHYS_SIDE_TRACKING_QUEUE_1300222049
#SBATCH --output=/allen/programs/mindscope/production/openscope/prod0/specimen_1294074615/ecephys_session_1300222049/202310281932_ECEPHYS_SIDE_TRACKING_QUEUE_1300222049.log
#SBATCH --chdir=/allen/programs/mindscope/production/openscope/prod0/specimen_1294074615/ecephys_session_1300222049/
#SBATCH --mail-type=NONE 
umask 022
/allen/aibs/technology/conda/run_miniconda.sh /allen/aibs/technology/conda/production/dlcPy36 python /allen/aibs/technology/waynew/eye/bin/side_dlc_phase1.py /allen/programs/mindscope/production/openscope/prod0/specimen_1294074615/ecephys_session_1300222049/1306934652/1300222049_692077_20230928.behavior.mp4 /allen/programs/mindscope/production/openscope/prod0/specimen_1294074615/ecephys_session_1300222049/side_tracking/1300222049_692077_20230928.behaviorDeepCut_resnetNone_np3_side_camMar31shuffle1_1030000.h5
rtn_code=$?
curl --request GET -d return_code=$rtn_code http://lims2/jobs/finish_job/1307223802 --max-time 300 --retry 5 --retry-delay 300
exit $rtn_code
