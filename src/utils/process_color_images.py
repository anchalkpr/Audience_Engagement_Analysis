import os
from PIL import Image
import numpy as np
import face_recognition
from os.path import basename

from pandas._libs.parsers import basestring

# the data set contains quite a few color images that need to be converted to grey scale
# and resized to 100 x 100 to standardize all images
# this piece of code achieves that

# Original Dataset - https://github.com/e-drishti/wacv2016/tree/master/dataset

PATH_DATASET="/Users/manal/Downloads/wacv2016-master/dataset/3"

file_list = os.listdir(PATH_DATASET)

count=0
unable=0
for file in file_list:
    full_path = os.path.join(PATH_DATASET, file)
    image = np.array(Image.open(full_path))
    if image.shape != (100,100):
        face_location = face_recognition.face_locations(image)
        if(len(face_location) > 0):
            top, right, bottom, left = face_location[0]
            face_image = image[top:bottom, left:right]
            resized_image = Image.fromarray(face_image).convert('L').resize((100, 100))
            resized_image.save(os.path.splitext(full_path)[0]+"_processed.jpg")
            count+=1
        else:
            unable+=1

print(count)
print(unable)
