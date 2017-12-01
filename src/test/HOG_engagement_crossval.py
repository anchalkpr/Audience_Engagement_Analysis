
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from skimage.feature import hog
from skimage import data, color, exposure
from PIL import Image
import matplotlib.pyplot as plt

import os
from random import shuffle

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif

from sklearn import svm
from sklearn import neural_network
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import metrics

from PIL import ImageDraw


# In[2]:


data_path = 'wacv2016/dataset'

person_list = ['11', '20', '23', '18', '2', '12', '13',
                 '19', '14', '22', '3', '8', '6', '17',
                 '4', '10', '5', '1', '16', '9', '7', '21']

classes = os.listdir(data_path)
if '.DS_Store' in classes:
    classes.remove('.DS_Store')

person_sets = set()

for category in classes:
    img_path = os.path.join(data_path, category)
    image_files = os.listdir(img_path)
    for img_file in image_files:
        person_id = img_file.split('_')[0][3:]
        person_sets.add(person_id)
        
dev_persons = ['1', '20', '14', '7', '9', '4',
               '6', '8', '3', '10', '5', '11',
               '18', '16', '12', '22', '23', '19']
test_persons = ['2', '13', '17', '21']

shuffle(dev_persons)


# ## Separating developed and test datasets

# In[3]:


xtest = list()
ytest = list()

xdev = list()
ydev = list()

for category in classes:
    img_path = os.path.join(data_path, category)
    image_files = os.listdir(img_path)
    if '.DS_Store' in image_files:
        image_files.remove('.DS_Store')
        print(len(image_files))
        
    for img_file in image_files:
        person_id = img_file.split('_')[0][3:]
        
        img = np.array(Image.open(os.path.join(img_path,img_file)))
        if img.shape != (100,100):
            continue
        fd1 = hog(img, orientations=9, pixels_per_cell=(5, 5),
                    cells_per_block=(2, 2), block_norm='L2')
        
        fd2 = hog(img, orientations=9, pixels_per_cell=(100, 100),
                    cells_per_block=(1, 1))
        
        fd = np.hstack((fd1,fd2))
        
        if category == '1':
            label = 0
        elif category == '2':
            break
        elif category == '3':
            label = 1
        
        #label = int(category)
        
        if person_id in dev_persons:
            xdev.append(fd)
            ydev.append(label)
        elif person_id in test_persons:
            xtest.append(fd)
            ytest.append(label)
            
xtest = np.array(xtest)
ytest = np.array(ytest)

xdev = np.array(xdev)
ydev = np.array(ydev)


# ## cross validation

# In[190]:


val_num = 3
fold_num = len(dev_persons)/val_num
c_range = np.array([x for x in [0.1, 1, 5, 25, 50]])
k_range = np.array([200*(x) for x in range(1,6)])

accuracies = np.zeros((2, int(fold_num), len(k_range), len(c_range)))
f1_scores = np.zeros((2, int(fold_num), len(k_range), len(c_range)))

for f in range(int(fold_num)):
    val_persons = dev_persons[f*val_num:(f*val_num)+val_num]
      
    xtrain = list()
    ytrain = list()
    
    xval = list()
    yval = list()
    
    for category in classes:
        img_path = os.path.join(data_path, category)
        image_files = os.listdir(img_path)
        if '.DS_Store' in image_files:
            image_files.remove('.DS_Store')

        for img_file in image_files:
            person_id = img_file.split('_')[0][3:]

            img = np.array(Image.open(os.path.join(img_path,img_file)))
            if img.shape != (100,100):
                continue
                
            fd1 = hog(img, orientations=9, pixels_per_cell=(5, 5),
                        cells_per_block=(2, 2), block_norm='L2')
        
            fd2 = hog(img, orientations=9, pixels_per_cell=(100, 100),
                        cells_per_block=(1, 1))

            fd = np.hstack((fd1,fd2))
            
            
            if category == '1':
                label = 0
            elif category == '2':
                break
            elif category == '3':
                label = 1
        
            #label = int(category)
                
            img = (img.flatten())
            if person_id in dev_persons:
                if person_id in val_persons:
                    xval.append(fd)
                    yval.append(int(label))
                else:
                    xtrain.append(fd)
                    ytrain.append(int(label))
            else:
                continue
                
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)

    # shuffle training data
    s = np.arange(xtrain.shape[0])
    np.random.shuffle(s)

    xtrain = xtrain[s]
    ytrain = ytrain[s]

    xval = np.array(xval)
    yval = np.array(yval)

    for k in range(len(k_range)):
        selectKBest = SelectKBest(mutual_info_classif, k=k_range[k])
        selectKBest.fit(xtrain, ytrain)
        train_input = selectKBest.transform(xtrain)
        val_input = selectKBest.transform(xval)
        for i in range(len(c_range)):
            #print(ytrain)
            linear_svm = svm.LinearSVC(C=c_range[i], class_weight='balanced', random_state=1)
            linear_svm.fit(train_input, ytrain)

            prediction = linear_svm.predict(train_input)
            train_acc = metrics.accuracy_score(ytrain, prediction)
            train_f1 = metrics.f1_score(ytrain, prediction, average='macro')
            train_conf_met = metrics.confusion_matrix(ytrain, prediction)

            prediction = linear_svm.predict(val_input)
            val_acc = metrics.accuracy_score(yval, prediction)
            val_f1 = metrics.f1_score(yval, prediction, average='macro')
            val_conf_met = metrics.confusion_matrix(yval, prediction)

            print('fold#{} n_components={}, C={}'.format(f+1, k_range[k], c_range[i]))
            print(train_conf_met)
            print(val_conf_met)
            
            accuracies[0][f][k][i] = train_acc
            accuracies[1][f][k][i] = val_acc
            f1_scores[0][f][k][i] = train_f1
            f1_scores[1][f][k][i] = val_f1


# In[191]:


avg_accuracies = np.zeros((2, len(k_range), len(c_range)))
avg_f1_scores = np.zeros((2, len(k_range), len(c_range)))

for i in range(2):
    for k in range(len(k_range)):
        for c in range(len(c_range)):
            avg_accuracies[i][k][c] = np.mean(accuracies[i,:,k,c])
            avg_f1_scores[i][k][c] = np.mean(f1_scores[i,:,k,c])


# In[192]:


ind = np.arange(len(k_range))
width = 0.12

fig, ax = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(15)

rects1 = ax.bar(ind, avg_accuracies[0,:,0], width)
rects2 = ax.bar(ind + width, avg_accuracies[0,:,1], width)
rects3 = ax.bar(ind + 2*width, avg_accuracies[0,:,2], width)
rects4 = ax.bar(ind + 3*width, avg_accuracies[0,:,3], width)
rects5 = ax.bar(ind + 4*width, avg_accuracies[0,:,4], width)



# add some text for labels, title and axes ticks
ax.set_ylim(0,1)
ax.set_ylabel('accuracy')
ax.set_title('training accuracy score')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(k_range)

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]),
          tuple(['C: {}'.format(c_range[i]) for i in range(len(c_range))]))
plt.show()


# In[193]:


fig, ax = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(15)

rects1 = ax.bar(ind, avg_f1_scores[0,:,0], width)
rects2 = ax.bar(ind + width, avg_f1_scores[0,:,1], width)
rects3 = ax.bar(ind + 2*width, avg_f1_scores[0,:,2], width)
rects4 = ax.bar(ind + 3*width, avg_f1_scores[0,:,3], width)
rects5 = ax.bar(ind + 4*width, avg_f1_scores[0,:,4], width)



# add some text for labels, title and axes ticks
ax.set_ylim(0,1)
ax.set_ylabel('f1-score')
ax.set_title('training f1 score')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(k_range)

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]),
          tuple(['C: {}'.format(c_range[i]) for i in range(len(c_range))]))
plt.show()


# In[194]:


fig, ax = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(15)

rects1 = ax.bar(ind, avg_accuracies[1,:,0], width)
rects2 = ax.bar(ind + width, avg_accuracies[1,:,1], width)
rects3 = ax.bar(ind + 2*width, avg_accuracies[1,:,2], width)
rects4 = ax.bar(ind + 3*width, avg_accuracies[1,:,3], width)
rects5 = ax.bar(ind + 4*width, avg_accuracies[1,:,4], width)

# add some text for labels, title and axes ticks
ax.set_ylim(0,1)
ax.set_ylabel('accuracy')
ax.set_title('validation accuracy score')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(k_range)

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]),
          tuple(['C: {}'.format(c_range[i]) for i in range(len(c_range))]))
plt.show()


# In[195]:


fig, ax = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(15)

rects1 = ax.bar(ind, avg_f1_scores[1,:,0], width)
rects2 = ax.bar(ind + width, avg_f1_scores[1,:,1], width)
rects3 = ax.bar(ind + 2*width, avg_f1_scores[1,:,2], width)
rects4 = ax.bar(ind + 3*width, avg_f1_scores[1,:,3], width)
rects5 = ax.bar(ind + 4*width, avg_f1_scores[1,:,4], width)



# add some text for labels, title and axes ticks
ax.set_ylim(0,1)
ax.set_ylabel('f1-score')
ax.set_title('validation f1 score')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(k_range)

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]),
          tuple(['C: {}'.format(c_range[i]) for i in range(len(c_range))]))
plt.show()


# In[196]:


print(np.argsort(avg_accuracies[1].flatten()))
print(avg_accuracies[1])


# In[197]:


print(np.argsort(avg_f1_scores[1].flatten()))
print(avg_f1_scores[1])