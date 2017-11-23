import csv
import numpy as np


def replace_yes_no(l):
    new_l = list()
    for x in l:
        if x == 'Yes':
            new_l.append(1)
        elif x == 'No':
            new_l.append(0)
        else:
            new_l.append(int(x))
    return new_l

responses = list()
header= list()
with open("/Users/manal/Downloads/Engagement Analysis - Evaluation (Responses) - Form Responses 1.csv") as infile:
    reader = csv.reader(infile)
    header = next(reader)
    header = header[2:]
    for row in reader:
        responses.append(replace_yes_no(row[2:]))

responses = np.array(responses)
responses_averaged = list()
start = 0
i = 1
image_counter = 1
while(i < len(header)):
    if 'Engaged?' in header[i]:
        i+=1
    else:
        print("Image "+str(image_counter)+": Start - "+str(start)+"; End - "+str(i))
        responses_for_an_image = responses[:, start:i+1]
        responses_avg_image = np.mean(responses_for_an_image, axis=0)
        responses_averaged.append(responses_avg_image)
        start = i+1
        i = start+1
        image_counter+=1

responses_averaged = np.array(responses_averaged)
#print(responses_averaged)
diff = 0
for i in range(len(responses_averaged)):
    row = responses_averaged[i]
    engagement_mean = np.mean(row[:-1])
    overall_engagement_scales = row[-1:][0]/10
    diff += np.abs(engagement_mean - overall_engagement_scales)
    print("Image "+str(i+1))
    print("Engagement mean (based on per person response): "+str(engagement_mean))
    print("Overall Engagement: "+str(overall_engagement_scales))
    print()

diff = diff/len(responses_averaged)
print("Averaged Variation: "+str(diff))




