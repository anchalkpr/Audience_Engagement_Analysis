import realtime.comman_utils as comman_utils
import os, time

#### CONSTANTS ##############
path_capture_dir = comman_utils.PATH_CAPTURE_DIR
path_output_dir = comman_utils.PATH_ANALYSIS_DIR
DEBUG = comman_utils.DEBUG
#############################

current_milli_time = lambda: int(round(time.time() * 1000))

# Cleanup output directory
comman_utils.clean_dir(path_output_dir)

os.makedirs(path_output_dir)














