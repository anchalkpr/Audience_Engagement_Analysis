## Detects the faces in the first frame
## and stores the cropped image for these faces in the output directory for the subsequent frames

import face_recognition
import cv2
import os
import time
import realtime.comman_utils as comman_utils

####### CONSTANTS #################
path_output_dir = comman_utils.PATH_CAPTURE_DIR
UNKNOWN = "unknown"
DEBUG = comman_utils.DEBUG
###################################

current_milli_time = lambda: int(round(time.time() * 1000))

# Cleanup output directory
comman_utils.clean_dir(path_output_dir)

os.makedirs(path_output_dir)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# Define the fps to be equal to 10. Also frame size is passed.

capture_video_out = cv2.VideoWriter(comman_utils.PATH_CAPTURE_VIDEO, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 480))

input("Press Enter when you are ready")
ret, init_frame = video_capture.read()
capture_video_out.write(init_frame)

init_frame_face_locations = face_recognition.face_locations(init_frame)
faceid_list = list()
face_encodings_list = list()

init_file_output_dir = os.path.join(path_output_dir, "init")
os.makedirs(init_file_output_dir)
#store faces in a file, find encodings
for i in range(len(init_frame_face_locations)):
    face_location = init_frame_face_locations[i]
    top, right, bottom, left = face_location

    face_image = init_frame[top:bottom, left:right]
    face_id = "face_" + str(i)
    path_file = os.path.join(init_file_output_dir, face_id + ".png")

    cv2.imwrite(path_file, face_image)
    face_encodings_list.append(face_recognition.face_encodings(face_image)[0])
    faceid_list.append(face_id)

process_this_frame = True
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    #write frame to output file
    capture_video_out.write(frame)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_name_list = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match_list = face_recognition.compare_faces(face_encodings_list, face_encoding)
            name = UNKNOWN

            for i in range(len(match_list)):
                if match_list[i] and faceid_list[i] not in face_name_list:
                    name = faceid_list[i]
                    break

            face_name_list.append(name)

        # store all the faces for this frame in this directory
        current_output_dir = os.path.join(path_output_dir, str(current_milli_time()))
        os.makedirs(current_output_dir)

    process_this_frame = not process_this_frame

    unknown_counter = 0
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_name_list):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4; right *= 4; bottom *= 4; left *= 4
        if name is UNKNOWN:
            name = name + "_" +str(unknown_counter)
            unknown_counter+=1
        face_image = frame[top:bottom, left:right]
        path_file = os.path.join(current_output_dir, name + ".png")
        cv2.imwrite(path_file, face_image)

        if DEBUG:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    if DEBUG:
        # Display the resulting image
        cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam, output file
video_capture.release()
capture_video_out.release()
cv2.destroyAllWindows()