#!/usr/bin/env python
# coding: utf-8

# This is final clean code of the NNLS course project. 

# In[ ]:


import tensorflow as tf
from skimage.io import imread
from skimage.io import imsave
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import matplotlib.pyplot as plt
import skimage
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from google.colab import drive
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten


# In[ ]:





# In[ ]:





# In[ ]:


drive.mount('/content/drive')


# In[ ]:


# First check accuracy using our simple model on the complete mammogram dataset 
image_path = 'drive/My Drive/BreastCancer/NEW_COMPLETE_MAMMOGRAM'     # The original 215 images 

def loadImages(path,string):
    if(string=="neg"):
      p = 'negative_images'
    else:
      p = 'positive_images'
    image_files = sorted([os.path.join(path, p, file)
                          for file in os.listdir(path + "/"+p)
                          if file.endswith('.png')])
    return image_files


global image_path
negative_dataset = loadImages(image_path,"neg")
positive_dataset = loadImages(image_path,"pos")


# negative class 
for i in range(len(negative_dataset)):
  if(i==0):
    img = cv2.imread(negative_dataset[i], cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (299, 299)) 
    train_images = np.array([img])
    train_images_neg = np.array([img])
    train_labels = np.array([1,0])
  else:
    img = cv2.imread(negative_dataset[i], cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (299, 299)) 
    train_images = np.vstack((train_images,np.array([img])))
    train_images_neg = np.vstack((train_images_neg,np.array([img])))
    train_labels = np.vstack((train_labels,[1,0]))
    

# positive class 
for i in range(len(positive_dataset)):
  if(i==0):
    img = cv2.imread(positive_dataset[i], cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (299, 299)) 
    train_images_pos = np.array([img])
  img = cv2.imread(positive_dataset[i], cv2.IMREAD_GRAYSCALE)
  
  img = cv2.resize(img, (299, 299)) 
  train_images = np.vstack((train_images,np.array([img])))
  train_images_pos = np.vstack((train_images_pos,np.array([img])))
  train_labels = np.vstack((train_labels,[0,1]))
orig_train_images = train_images.copy()


# In[ ]:


# Do mean subtraction for each of the concerned image first 
for i,p in enumerate(orig_train_images):
  if(i==0):
    q = (p-np.mean(p))/np.std(p)
    orig_train_images_new = np.array([q])
  else:
    q = (p-np.mean(p))/np.std(p)
    orig_train_images_new = np.vstack((orig_train_images_new,[q]))
print(orig_train_images_new.shape)         # mean subtracted image 


# In[ ]:


train_images = orig_train_images_new 


# In[ ]:





# In[ ]:


# resize the train images so loaded 

print(train_images.shape)
train_images = np.resize(train_images,(train_images.shape[0],299,299,1))  # Required for CNN
print(train_images.shape)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.20, random_state=42,stratify = train_labels, shuffle=True)
print(X_train.shape)


# In[ ]:


# print(orig_train_images.shape)
print(X_test.shape)
print(X_train.shape)
# X_temp = X_train.copy()
# X_temp = np.repeat(X_temp,3,axis=3)
# This is for Resnet 

X_train_resnet = np.repeat(X_train,3,axis=3)
X_test_resnet = np.repeat(X_test,3,axis=3)

print(X_train_resnet.shape)


# In[ ]:





# In[ ]:


# # Let's try data generation here 
# for i in range(train_images_pos.shape[0]):
#   image = train_images_pos[i]
#   image = np.expand_dims(image, 2)
#   image = np.expand_dims(image, 0)
#   path = 'drive/My Drive/BreastCancer/NEW_COMPLETE_MAMMOGRAM/generated_data/positive_images/'
#   datagen = ImageDataGenerator(rotation_range=60, width_shift_range=0.1, height_shift_range=0.1,shear_range=0.15, zoom_range=0.5,channel_shift_range = 10, horizontal_flip=True)
#   datagen.fit(image)
#   count = 0 
#   for x, val in zip(datagen.flow(image),range(10)) :  
#     x = np.resize(x,(x.shape[1],x.shape[2]))
#     x = x.astype('uint8') 
#     imsave(path+str(10*i+count)+'.png',x)
#     count+=1
# for i in range(train_images_neg.shape[0]):
#   image = train_images_neg[i]
#   image = np.expand_dims(image, 2)
#   image = np.expand_dims(image, 0)
#   path = 'drive/My Drive/BreastCancer/NEW_COMPLETE_MAMMOGRAM/generated_data/negative_images/'
#   datagen = ImageDataGenerator(rotation_range=60, width_shift_range=0.1, height_shift_range=0.1,shear_range=0.15, zoom_range=0.5,channel_shift_range = 10, horizontal_flip=True)
#   datagen.fit(image)
#   count = 0 
#   for x, val in zip(datagen.flow(image),range(10)) :  
#     x = np.resize(x,(x.shape[1],x.shape[2]))
#     x = x.astype('uint8') 
#     imsave(path+str(10*i+count)+'.png',x)
#     count+=1


# In[ ]:


# complete mammogram classification model
# model = models.Sequential()
# model.add(layers.Conv2D(64, (3, 3), input_shape=(299,299,1)))
# model.add(layers.BatchNormalization())
# model.add(layers.ReLU())
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(32, (3, 3)))
# model.add(layers.BatchNormalization())
# model.add(layers.ReLU())
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(32, (3, 3)))
# model.add(layers.BatchNormalization())
# model.add(layers.ReLU())
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(16, (3, 3)))
# model.add(layers.BatchNormalization())
# model.add(layers.ReLU())
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(16, (3, 3)))
# model.add(layers.BatchNormalization())
# model.add(layers.ReLU())
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(299,299,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))


# model = Sequential()
# model.add(Conv2D(input_shape=(299,299,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


# In[ ]:


model.add(layers.Flatten())
model.add(Dense(128, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(layers.Dense(2))


# In[ ]:


model.summary()


# In[ ]:


# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
ACCURACY_THRESHOLD = 0.85
class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_acc = logs.get('val_accuracy')
        acc = logs.get('accuracy')
        if(val_acc >= self.threshold and acc>=self.threshold):
            self.model.stop_training = True

callback = MyThresholdCallback(ACCURACY_THRESHOLD)


# In[ ]:


es = EarlyStopping(monitor='val_loss', patience = 3, verbose=1,restore_best_weights=True)


# In[ ]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50,validation_data=(X_test[:32], y_test[:32]),callbacks=[es])


# In[ ]:


test_loss, test_acc = model.evaluate(X_test[32:],  y_test[32:], verbose=2)


# In[ ]:


import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, 'r', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['accuracy']
val_loss_values = history_dict['val_accuracy']

epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, 'r', label='Training Accuracy')
plt.plot(epochs, val_loss_values, 'b', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:



print(y_test)


# In[ ]:


es = EarlyStopping(monitor='val_loss', patience = 3, verbose=1,restore_best_weights=True)


# In[ ]:


# Let's try Resnet on the complete mammogram first # Note that images are mean scaled  

from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras
# restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(299,299,3))
# output = restnet.layers[-1].output
# out = keras.layers.Flatten()(output)
# restnet = Model(restnet.input, output=out)
# for layer in restnet.layers:
#   layer.trainable = False
# restnet.summary()
resmodel = ResNet50(input_shape=(299,299,3), weights='imagenet', include_top=False)

# don't train existing weights
for layer in resmodel.layers:
  layer.trainable = False

x = Flatten()(resmodel.output)
# x = Dense(1000, activation='relu')(x)
initializer = tf.keras.initializers.GlorotNormal()
y = Dense(256, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),kernel_initializer=initializer)(x)
prediction = Dense(2, activation='softmax')(y)

# create a model object
model = Model(inputs=resmodel.input, outputs=prediction)
# model.summary()


# In[ ]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train_resnet, y_train, epochs=50,validation_data=(X_test_resnet[:X_test_resnet.shape[0]*2//3], y_test[:X_test_resnet.shape[0]*2//3]),callbacks=[es])


# In[ ]:


test_loss, test_acc = model.evaluate(X_test_resnet[X_test_resnet.shape[0]*2//3:],  y_test[X_test_resnet.shape[0]*2//3:], verbose=2)


# In[ ]:


# Plots for training and validation accuracy 
import matplotlib.pyplot as plt
# print(X_test_resnet.shape)
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, 'r', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['accuracy']
val_loss_values = history_dict['val_accuracy']

epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, 'r', label='Training Accuracy')
plt.plot(epochs, val_loss_values, 'b', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


# Let's test on some unseen data 
model.summary()


# In[ ]:


from google.colab.patches import cv2_imshow


# In[ ]:


model = keras.models.load_model("ResNetCompleteMammogram.h5")
image_1 = cv2.imread('image_1.png')               # This is very import either use skimage OR cv2, dont use both 
image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGRA2GRAY)
print(image_1.shape)
image_1 = cv2.resize(image_1,(299,299))

image_1 = np.resize(image_1,(1,image_1.shape[0],image_1.shape[1],1))
print(image_1.shape)
image_1 = np.repeat(image_1,3,axis=3)
prediction = model.predict(image_1)
print(prediction)                              # Good prediction 


# In[ ]:


# model = keras.models.load_model("ResNetCompleteMammogram.h5")
image_1 = cv2.imread('image_2.jpeg')               # This is very import either use skimage OR cv2, dont use both 

print(image_1.shape)
image_1 = cv2.resize(image_1,(299,299))

image_1 = np.resize(image_1,(1,image_1.shape[0],image_1.shape[1],1))
print(image_1.shape)
image_1 = np.repeat(image_1,3,axis=3)
prediction = model.predict(image_1)
print(prediction)                              # Good prediction again  


# In[ ]:


image_1 = cv2.imread('image_3.png')               # This is very import either use skimage OR cv2, dont use both 

print(image_1.shape)
image_1 = cv2.resize(image_1,(299,299))

image_1 = np.resize(image_1,(1,image_1.shape[0],image_1.shape[1],1))
print(image_1.shape)
image_1 = np.repeat(image_1,3,axis=3)
prediction = model.predict(image_1)
print(prediction)                              # Good prediction again  


# In[ ]:


image_1 = cv2.imread('image_4.png')               # This is very import either use skimage OR cv2, dont use both 
cv2_imshow(image_1)
print(image_1.shape)
image_1 = cv2.resize(image_1,(299,299))

image_1 = np.resize(image_1,(1,image_1.shape[0],image_1.shape[1],1))
print(image_1.shape)
image_1 = np.repeat(image_1,3,axis=3)
prediction = model.predict(image_1)
print(prediction)                              # Good prediction again  


# In[ ]:


# Save weights of this model 

model.save('drive/My Drive/BreastCancer/Model Weights/ResNetCompleteMammogram.h5')


# In[ ]:


from google.colab import files
files.download("ResNetCompleteMammogram.h5")          # Download the Weights


# In[ ]:


# ROI classification model Let's hope it works well  
# Simple CNN model would be tested and is hoped to work well here after that we can save the model weights 

image_path = 'drive/My Drive/BreastCancer/ROI_Images/AUG_ROI_IMAGES'


# In[ ]:


def loadImages(path,string):
    if(string=="neg"):
      p = 'negative_images'
    else:
      p = 'positive_images'
    image_files = sorted([os.path.join(path, p, file)
                          for file in os.listdir(path + "/"+p)
                          if file.endswith('.png')])
    return image_files

global image_path
negative_dataset = loadImages(image_path,"neg")
positive_dataset = loadImages(image_path,"pos")
print(len(negative_dataset))
print(len(positive_dataset))


# In[ ]:


total_dataset = negative_dataset + positive_dataset
# Now store all the images in the total dataset

for i in range(len(total_dataset)):
  if(i<len(negative_dataset)):
    if(i==0):
      train_labels = np.array([1,0])
    else:
      train_labels = np.vstack((train_labels,[1,0]))
  else:
    train_labels = np.vstack((train_labels,[0,1]))

  if(i==0):
    train_images = np.array([cv2.imread(total_dataset[i], cv2.IMREAD_GRAYSCALE)])
    
  else:
    img = cv2.imread(total_dataset[i], cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))             # resize just to be sure here 
    train_images = np.vstack((train_images,np.array([img])))
    
print(train_images.shape)  
# Now we have loaded all the images into an array  cheap 


# In[ ]:


print(train_images.shape)


# In[ ]:


# Now do mean centering and standardization here 
# We will do x = (x-mean)/std_deviation, the visual perceptibility might be lost 

def mean_centering_standardization(train_images):
  for i in range(len(total_dataset)):
    if(i==0):
      img = train_images[i]
      img_new = (img-np.mean(img))/np.std(img)
      trainset_centered = np.array([img_new])
    else:
      img = train_images[i]
      img_new = (img-np.mean(img))/np.std(img)
      trainset_centered = np.vstack((trainset_centered,np.array([img_new])))

  return trainset_centered

trainset_centered = mean_centering_standardization(train_images)
train_images = trainset_centered
# Now we have our mean centered and standardized image set 


# In[ ]:


print(train_images.shape)
train_images = np.resize(train_images,(train_images.shape[0],100,100,1))
print(train_images.shape)


# In[ ]:


# Shuffle the Dataset and divide into training and test set 
 
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.20, random_state=101,stratify = train_labels, shuffle=True)


# In[ ]:


print(y_test.shape)


# In[ ]:


# A Simple CNN Model ::
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(100,100,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))



# In[ ]:


# Model Summary
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2,activation='softmax'))

model.summary()


# In[ ]:


early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)


# In[ ]:


# Fitting it 
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20,validation_data=(X_test[:X_test.shape[0]*2//3], y_test[:X_test.shape[0]*2//3]), callbacks=[early])         #,callbacks=[es]


# In[ ]:


test_loss, test_acc = model.evaluate(X_test[X_test.shape[0]*2//3:],  y_test[X_test.shape[0]*2//3:], verbose=2)   # Good accuracy on the test set also


# In[ ]:


# Now see the plots here 
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, 'r', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['accuracy']
val_loss_values = history_dict['val_accuracy']

epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, 'r', label='Training Accuracy')
plt.plot(epochs, val_loss_values, 'b', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


# Save the weights 
model.save('drive/My Drive/BreastCancer/Model Weights/ROICNN_final_weights.h5')


# In[ ]:


from google.colab import files
files.download('ROICNN_final_weights.h5')


# In[ ]:


# Automatic detection of ROI in complete mammogram # See Roshan's Work here  

def detect_anamoly(mammogram):            # Detect anamoly in this mammogram 
  # First check whether the mammogram is positive threshold it around ~0.35
  # For ResNet based models you have to do some preprocessing here 


  # If greater than 0.5 then ofcourse there's a cancerous element
  # Here use that simple ROI model 


  # If uncertainity prevails then still try to detect lesions 
  # Here also use that simple ROI model 


  # If certainly NO then output so

  print("Model is quite certain that no observable lesion exists")

  return 
  
detect_anamoly(mammogram_image) 


# In[ ]:


model.summary()


# In[ ]:


get_ipython().system('pip install flask-ngrok')


# In[ ]:





# In[ ]:





# In[ ]:


# Let's plot the confusion matrix now for the test dataset only and display the results as percentage 
from sklearn.metrics import confusion_matrix
resmodel = keras.models.load_model('drive/My Drive/BreastCancer/Model Weights/ResNetCompleteMammogram.h5')
roimodel = keras.models.load_model('drive/My Drive/BreastCancer/Model Weights/ROICNN_final_weights.h5')


# In[ ]:


predictions = resmodel.predict(X_test_resnet)
print(predictions.shape)


# In[ ]:


predictions_conf = np.argmax(predictions, axis=1)
print(predictions_conf)


# In[ ]:


ytest_conf = np.argmax(y_test, axis=1)           # This y_test is for the complete mammogram 
print(ytest_conf)


# In[ ]:


conf_matrix = confusion_matrix(ytest_conf, predictions_conf)
print(conf_matrix)
tn, fp, fn, tp = conf_matrix.ravel()
print(tn,fp,fn,tp)


# In[ ]:


# Now let's try to do it for the ROI model-dataset
predictions = roimodel.predict(X_test)              # Predictions 
print(predictions.shape)


# In[ ]:


predictions_conf = np.argmax(predictions, axis=1)
print(predictions_conf)


# In[ ]:


ytest_conf = np.argmax(y_test, axis=1)           # This y_test is for the complete mammogram 
print(ytest_conf)


# In[ ]:


conf_matrix = confusion_matrix(ytest_conf, predictions_conf)
print(conf_matrix)
tn, fp, fn, tp = conf_matrix.ravel()
print(tn,fp,fn,tp)

