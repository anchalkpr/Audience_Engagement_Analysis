import comman_utils as comman_utils
import os, time, sys, signal
import subprocess
import cv2
import  au_detection as au_detection

#### CONSTANTS ##############
path_capture_dir = comman_utils.PATH_CAPTURE_DIR
path_output_dir = comman_utils.PATH_ANALYSIS_DIR
path_openface_featureextraction = os.path.join(comman_utils.PATH_OPENFACE_BIN, "FeatureExtraction")
DEBUG = comman_utils.DEBUG
FNULL = open(os.devnull, 'w')
#############################


def run_featureextraction(capture_dir_name):
    subprocess.call(
        [path_openface_featureextraction,
         '-au_static',
         '-fdir', os.path.join(path_capture_dir, capture_dir_name),
         '-of', os.path.join(path_output_dir, capture_dir_name + '.csv'),
         '-no2Dfp', '-no3Dfp', '-noMparams', '-noPose', '-q'],
        stdout=FNULL)

def analyze_face_main():
    print("Generating AU files")
    current_milli_time = lambda: int(round(time.time() * 1000))

    # Cleanup output directory
    comman_utils.clean_dir(path_output_dir)

    if not os.path.exists(path_output_dir):
        os.makedirs(path_output_dir)

    # wait till init directory is created by facial recognizer and tracker
    while os.path.exists(os.path.join(path_capture_dir, comman_utils.INIT_DIR_NAME)) is False:
        time.sleep(1)

    # process init directory
    run_featureextraction(comman_utils.INIT_DIR_NAME)
    key = cv2.waitKey(0) & 0b11111111

    dir_list = sorted(os.listdir(path_capture_dir))
    for dir in dir_list:
        run_featureextraction(dir)

    au_detection.au_calculation()




