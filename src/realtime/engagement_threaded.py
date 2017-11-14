## Detects the faces in the first frame
## and stores the cropped image for these faces in the output directory for the subsequent frames

import face_recognition, cv2
import os, threading, time
import queue
import realtime.comman_utils as comman_utils
from sklearn.externals import joblib
from PIL import Image
import numpy as np
import datetime

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

####### CONSTANTS #################
path_output_dir = comman_utils.PATH_CAPTURE_DIR
UNKNOWN = "unknown"
DEBUG = comman_utils.DEBUG
ENGAGEMENT_CHART_XAXIS_TIME_LIMIT_MIN = 10
###################################

######## Global Variables ###############
Frame_Queue = queue.Queue()
Frame_Face_ImageList_Queue = queue.Queue()
Engagement_X_Axis_List = list()
Engagement_Y_Axis_List = list()
#########################################

captureTime = input("How long you wanna capture video for:");
current_milli_time = lambda: int(round(time.time() * 1000))

# Cleanup output directory
comman_utils.clean_dir(path_output_dir)

if not os.path.exists(path_output_dir):
    os.makedirs(path_output_dir)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

#FRAME_RATE = video_capture.get(cv2.CAP_PROP_FPS)
## Define process rate for frames
#process_rate = math.ceil(FRAME_RATE / 15)

height, width, channels = video_capture.read()[1].shape
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# Define the fps to be equal to 10. Also frame size is passed.
capture_video_out = cv2.VideoWriter(comman_utils.PATH_CAPTURE_VIDEO, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (width, height))

######## Start Recording Video ###############

'''
input("Press Enter when you are ready")
ret, init_frame = video_capture.read()
capture_video_out.write(init_frame)

init_frame_face_locations = face_recognition.face_locations(init_frame)
faceid_list = list()
face_encodings_list = list()

path_init_dir = os.path.join(path_output_dir, comman_utils.INIT_DIR_NAME)
os.makedirs(path_init_dir)
#store init faces in a file, find encodings
for i in range(len(init_frame_face_locations)):
    face_location = init_frame_face_locations[i]
    top, right, bottom, left = face_location

    face_image = init_frame[top:bottom, left:right]
    face_id = "face_" + str(i)
    path_file = os.path.join(path_init_dir, face_id + ".png")

    cv2.imwrite(path_file, face_image)
    face_encodings_list.append(face_recognition.face_encodings(face_image)[0])
    faceid_list.append(face_id)
    '''

# function to read frames from the webcam and add it to the queue
def capture_video():
    t = threading.currentThread()
    frame_counter = 0
    frames_captured = 0
    while getattr(t, "do_run", True):
        ret, frame = video_capture.read()
        if ret and frame_counter%2==0:
            Frame_Queue.put((current_milli_time(), frame))
            frames_captured+=1
        frame_counter+=1
    print("Frames captured: "+str(frames_captured))


# function to process frames
def process_frames():
    t = threading.currentThread()
    frames_processed_counter = 0
    while getattr(t, "do_run", True):
        try:
            timestamp, frame = Frame_Queue.get(timeout=1)
            frames_processed_counter+=1
            # write frame to output file
            capture_video_out.write(frame)

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Find all the faces in the current frame of video
            face_locations = face_recognition.face_locations(small_frame)
            face_image_list = []
            for top, right, bottom, left in face_locations:
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4;
                right *= 4;
                bottom *= 4;
                left *= 4
                face_image = frame[top:bottom, left:right]
                face_image_list.append((face_image, (top, bottom, left, right)))
            Frame_Face_ImageList_Queue.put((timestamp, face_image_list, frame))
        except queue.Empty:
            pass
    print("Frames processed: " + str(frames_processed_counter))

#function to determine engagement level
def determine_engagement():
    model = joblib.load(comman_utils.PATH_ENGAGEMENT_MODEL)
    pca = joblib.load(comman_utils.PATH_PCA_MODEL)
    t = threading.currentThread()
    frame_faces_processed = 0
    while getattr(t, "do_run", True):
        try:
            timestamp, face_image_list, frame = Frame_Face_ImageList_Queue.get(timeout=1)
            frame_faces_processed+=1
            face_image_transformed_list = []
            location_list = []
            for face_image, top_bottom_left_right in face_image_list:
                img = np.array(Image.fromarray(face_image).convert('L').resize((100, 100))).flatten()
                img = (img / 255.0)
                face_image_transformed_list.append(img)
                location_list.append(top_bottom_left_right)
            reduced_x = pca.transform(face_image_transformed_list)
            prediction = model.predict(reduced_x)
            engagement = np.mean(prediction)

            for engagement_level, top_bottom_left_right in zip(prediction, location_list):
                top = top_bottom_left_right[0]
                bottom = top_bottom_left_right[1]
                left = top_bottom_left_right[2]
                right = top_bottom_left_right[3]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, str(engagement_level), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.imshow('Video', frame)
            Engagement_Y_Axis_List.append(engagement)
            Engagement_X_Axis_List.append(datetime.datetime.fromtimestamp(timestamp / 1000.0))
        except:
            pass
    print("Frame (for engagement) processed: " + str(frame_faces_processed))

# display engagement levels in a graph
def display_engagement_level(i):
    # remove old content
    # current_time = current_milli_time()
    # for i in range(len(Engagement_X_Axis_List)):
    #     if (datetime.datetime.fromtimestamp(current_time/1000) - Engagement_X_Axis_List[i]) \
    #             < datetime.timedelta(minutes=ENGAGEMENT_CHART_XAXIS_TIME_LIMIT_MIN):
    #         break
    # Engagement_X_Axis_List = Engagement_X_Axis_List[i:]
    # Engagement_Y_Axis_List = Engagement_Y_Axis_List[i:]
    sub_plot.clear()
    sub_plot.plot(Engagement_X_Axis_List, Engagement_Y_Axis_List)

cv2.namedWindow("Video")
# start time
start_processing_time_ms = current_milli_time()

# Start a thread to capture video from the webcam
capture_video_thread = threading.Thread(target=capture_video)
capture_video_thread.start()

#start a thread to process the frames
process_frames_thread = threading.Thread(target=process_frames)
process_frames_thread.start()

#start a thread to determine the engagement level
determine_engagement_thread = threading.Thread(target=determine_engagement)
determine_engagement_thread.start()

#display engagement levels
style.use('fivethirtyeight')
plot_figure = plt.figure()
sub_plot = plot_figure.add_subplot(1,1,1)
ani = animation.FuncAnimation(plot_figure, display_engagement_level, interval=1000)
plt.show()

while (float(captureTime) > (current_milli_time() - start_processing_time_ms)/1000):
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
        time.sleep(0.5)

#kill engagement thread
determine_engagement_thread.do_run = False
determine_engagement_thread.join()

#kill process frame thread
process_frames_thread.do_run = False
process_frames_thread.join()

#kill capture video thread
capture_video_thread.do_run = False
capture_video_thread.join()

# Release handle to the webcam, output file
video_capture.release()
capture_video_out.release()
cv2.destroyAllWindows()

#Video captured.
print("Video captured. Execution time: "+str((current_milli_time()-start_processing_time_ms)/1000)+" seconds")

#call analyze_faces
#analyzeFace.analyze_face_main()