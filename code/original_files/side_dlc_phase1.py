#import tensorflow as tf
import os

os.environ["DLClight"]="True"
import sys

import deeplabcut

def main():
if __name__ == "__main__":
    main()
video_file_path = sys.argv[1]
output_file_path = sys.argv[2]
#output_file = video_file_path[:-4] + 'DeepCut_resnet50_universal_eye_trackingApr25shuffle1_969000.h5'
#output_file = video_file_path[:-4] + 'DeepCut_resnet50_universal_eye_trackingJul10shuffle1_1030000.h5'  #eye prod
output_file = video_file_path[:-4] + 'DeepCut_resnetNone_np3_side_camMar31shuffle1_1030000.h5' #face                           #WBW DEBUG TRY THE NEXT LINE HARDCODED
#output_file = '/allen/aibs/technology/waynew/allen/programs/braintv/production/visualbehavior/prod0/specimen_1109521187/ecephys_session_1131778697/face_tracking/wbw_output_file.h5'

# ### Path to trained model:
#path_config_file = '/allen/programs/braintv/workgroups/cortexmodels/peterl/visual_behavior/DLC_models/universal_eye_tracking-peterl-2019-04-25/config.yaml'
#path_config_file = '/allen/programs/braintv/workgroups/cortexmodels/peterl/visual_behavior/DLC_models/universal_eye_tracking-peterl-2019-07-10/config.yaml'
#path_config_file = '/allen/aibs/technology/waynew/eye/universal_eye_tracking-peterl-2019-07-10/config.yaml'
path_config_file = '/allen/aibs/technology/waynew/eye/np3_side_cam-sam_corbett-2020-03-31/config_ubuntu.yaml'

# ### Track points in video and generate h5 file:
#deeplabcut.analyze_videos(path_config_file,[video_file_path]) #can take a list of input videos
#deeplabcut.analyze_videos(path_config_file, [video_file_path], save_as_csv=False, videotype='.mp4', gputouse=cudadev, destfolder='/allen/aibs/technology/waynew/allen/programs/braintv/production/visualbehavior/prod0/specimen_1109521187/ecephys_session_1131778697/side_tracking/') #side version 2021
deeplabcut.analyze_videos(path_config_file, [video_file_path], save_as_csv=False, videotype='.mp4', gputouse=cudadev)

os.rename(output_file,output_file_path)  # WBW DEBUG SKIP THIS FOR NOW

# OPTIONAL: create videos with overlayed keypoint labels: this is optional but highly recommended for trouble-shooting. You can also save out specific labeled frames by setting the Frames2plot flag to specific frame indices, e.g. Frames2plot=[1,100,1000,10000,100000]:
#deeplabcut.create_labeled_video(path_config_file,video_file_path, videotype='.mp4')
