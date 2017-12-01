
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
from sklearn.manifold import TSNE

from sklearn import svm
from sklearn import neural_network
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import metrics

import pickle


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
        #print(len(image_files))
        
    for img_file in image_files:
        person_id = img_file.split('_')[0][3:]
        
        img = np.array(Image.open(os.path.join(img_path,img_file)))
        if img.shape != (100,100):
            continue
           
        if category == '1':
            label = 0
        elif category == '2':
            break
        elif category == '3':
            label = 1
        
        #label = int(category)
            
        img = (img.flatten())
        if person_id in dev_persons:
            xdev.append(img)
            ydev.append(label)
        elif person_id in test_persons:
            xtest.append(img)
            ytest.append(label)
            
xtest = np.array(xtest)
ytest = np.array(ytest)

xdev = np.array(xdev)
ydev = np.array(ydev)

ndev, nfeature = xdev.shape
ntest = xtest.shape[0]

# In[79]:


# global centering
dev_mean = xdev.mean(axis=0)
xdev = xdev - dev_mean
xtest = xtest - dev_mean

#local centering
xdev -= xdev.mean(axis=1).reshape(ndev, -1)
xtest -= xtest.mean(axis=1).reshape(ntest, -1)

# # Linear kernel SVM

# In[150]:


val_num = 2
fold_num = len(dev_persons)/val_num
c_range = np.array([0.02*(x) for x in range(1,8)])
p_range = np.array([50*(x) for x in range(1,11)])

accuracies = np.zeros((2, int(fold_num), len(p_range), len(c_range)))
f1_scores = np.zeros((2, int(fold_num), len(p_range), len(c_range)))

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
                    xval.append(img)
                    yval.append(int(label))
                else:
                    xtrain.append(img)
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
    
    train_mean = xtrain.mean(axis=0)
    xtrain = xtrain - train_mean
    xval = xval - train_mean

    ntrain = xtrain.shape[0]
    nval = xval.shape[0]
    
    xtrain -= xtrain.mean(axis=1).reshape(ntrain, -1)
    xval -= xval.mean(axis=1).reshape(nval, -1)

    for p in range(len(p_range)):
        pca = PCA(n_components=p_range[p])
        pca.fit(xtrain)
        train_input = pca.transform(xtrain)
        val_input = pca.transform(xval)
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

            print('fold#{} n_components={}, C={}'.format(f+1, p_range[p], c_range[i]))
            print(train_conf_met)
            print(val_conf_met)
            
            accuracies[0][f][p][i] = train_acc
            accuracies[1][f][p][i] = val_acc
            f1_scores[0][f][p][i] = train_f1
            f1_scores[1][f][p][i] = val_f1

# In[151]:


avg_accuracies = np.zeros((2, len(p_range), len(c_range)))
avg_f1_scores = np.zeros((2, len(p_range), len(c_range)))

for i in range(2):
    for p in range(len(p_range)):
        for c in range(len(c_range)):
            avg_accuracies[i][p][c] = np.mean(accuracies[i,:,p,c])
            avg_f1_scores[i][p][c] = np.mean(f1_scores[i,:,p,c])


# In[152]:


ind = np.arange(len(p_range))
width = 0.12

fig, ax = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(15)

rects1 = ax.bar(ind, avg_accuracies[0,:,0], width)
rects2 = ax.bar(ind + width, avg_accuracies[0,:,1], width)
rects3 = ax.bar(ind + 2*width, avg_accuracies[0,:,2], width)
rects4 = ax.bar(ind + 3*width, avg_accuracies[0,:,3], width)
rects5 = ax.bar(ind + 4*width, avg_accuracies[0,:,4], width)
rects6 = ax.bar(ind + 5*width, avg_accuracies[0,:,5], width)
rects7 = ax.bar(ind + 6*width, avg_accuracies[0,:,6], width)



# add some text for labels, title and axes ticks
ax.set_ylim(0,1)
ax.set_ylabel('accuracy')
ax.set_title('training accuracy score')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(p_range)

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0], rects7[0]),
          tuple(['C: {}'.format(c_range[i]) for i in range(len(c_range))]))
plt.show()


# In[153]:


fig, ax = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(15)

rects1 = ax.bar(ind, avg_f1_scores[0,:,0], width)
rects2 = ax.bar(ind + width, avg_f1_scores[0,:,1], width)
rects3 = ax.bar(ind + 2*width, avg_f1_scores[0,:,2], width)
rects4 = ax.bar(ind + 3*width, avg_f1_scores[0,:,3], width)
rects5 = ax.bar(ind + 4*width, avg_f1_scores[0,:,4], width)
rects6 = ax.bar(ind + 5*width, avg_f1_scores[0,:,5], width)
rects7 = ax.bar(ind + 6*width, avg_f1_scores[0,:,6], width)



# add some text for labels, title and axes ticks
ax.set_ylim(0,1)
ax.set_ylabel('f1-score')
ax.set_title('training f1 score')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(p_range)

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0], rects7[0]),
          tuple(['C: {}'.format(c_range[i]) for i in range(len(c_range))]))
plt.show()


# In[154]:


fig, ax = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(15)

rects1 = ax.bar(ind, avg_accuracies[1,:,0], width)
rects2 = ax.bar(ind + width, avg_accuracies[1,:,1], width)
rects3 = ax.bar(ind + 2*width, avg_accuracies[1,:,2], width)
rects4 = ax.bar(ind + 3*width, avg_accuracies[1,:,3], width)
rects5 = ax.bar(ind + 4*width, avg_accuracies[1,:,4], width)
rects6 = ax.bar(ind + 5*width, avg_accuracies[1,:,5], width)
rects7 = ax.bar(ind + 6*width, avg_accuracies[1,:,6], width)

# add some text for labels, title and axes ticks
ax.set_ylim(0,1)
ax.set_ylabel('accuracy')
ax.set_title('validation accuracy score')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(p_range)

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0], rects7[0]),
          tuple(['C: {}'.format(c_range[i]) for i in range(len(c_range))]))
plt.show()


# In[155]:


fig, ax = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(15)

rects1 = ax.bar(ind, avg_f1_scores[1,:,0], width)
rects2 = ax.bar(ind + width, avg_f1_scores[1,:,1], width)
rects3 = ax.bar(ind + 2*width, avg_f1_scores[1,:,2], width)
rects4 = ax.bar(ind + 3*width, avg_f1_scores[1,:,3], width)
rects5 = ax.bar(ind + 4*width, avg_f1_scores[1,:,4], width)
rects6 = ax.bar(ind + 5*width, avg_f1_scores[1,:,5], width)
rects7 = ax.bar(ind + 6*width, avg_f1_scores[1,:,6], width)



# add some text for labels, title and axes ticks
ax.set_ylim(0,1)
ax.set_ylabel('f1-score')
ax.set_title('validation f1 score')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(p_range)

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0], rects7[0]),
          tuple(['C: {}'.format(c_range[i]) for i in range(len(c_range))]))
plt.show()


# In[156]:


print(np.argsort(avg_accuracies[1].flatten()))
print(avg_accuracies[1])


# In[157]:


print(np.argsort(avg_f1_scores[1].flatten()))
print(avg_f1_scores[1])
