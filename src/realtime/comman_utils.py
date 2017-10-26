import os
import shutil

DEBUG = True

PATH_CAPTURE_DIR = os.path.join("..", "..", "output", "capture")
PATH_CAPTURE_VIDEO = os.path.join("..", "..", "output", "capture_video.avi")
PATH_ANALYSIS_DIR = os.path.join("..", "..", "output", "analysis")
PATH_OPENFACE_BIN = "/Users/manal/Projects/Tools/OpenFace-master/bin"
INIT_DIR_NAME = "init"

def clean_dir(path):
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)