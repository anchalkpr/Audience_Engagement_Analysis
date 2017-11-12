# coding: utf-8

from sklearn.externals import joblib
model = joblib.load(model_file)
pca = joblib.load(pca_model_file)

# images are expected to be 100x100 size

# predicting the engagement for a single image
img = np.array(Image.open(image_file))
img = (img.flatten()/255.0)

reduced_x = pca.transform([img])
prediction = model.predict(reduced_x)


# predicting the engagement for a batch of images
img_list = []
for image_file in image_file_list:
    img = np.array(Image.open(image_file))
    img = (img.flatten()/255.0)
    img_list.append(img)
    
reduced_x = pca.transform(img_list)
prediction = model.prediction(reduced_x)