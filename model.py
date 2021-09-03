from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Lambda, Input, AveragePooling2D
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import random
import os

image_names=list(os.listdir("PROJECT_DATASET/"))
image_names.sort()
print(image_names)
class_number = len(image_names)


device = tf.test.gpu_device_name()
print(device)


data_dir = "PROJECT_DATASET/"
batch_size = 128
img_height, img_width = 224,224
epochs = 50


datagen= ImageDataGenerator(rescale=1/255,validation_split=.3,rotation_range=20,
                           shear_range=.2,width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2)

# Flow training images in batches of batch_size using train_data 
train_generator= datagen.flow_from_directory(
                data_dir,
                target_size=(img_width,img_height),
                batch_size=batch_size,
                subset="training",
                class_mode="categorical", 
                classes= image_names,
                shuffle=False,
                seed=30)


datagen2=ImageDataGenerator(rescale=1/255,validation_split=.3)

val_generator=datagen2.flow_from_directory(
                data_dir,
                target_size=(img_width,img_height),
                batch_size=batch_size,
                classes= image_names,
                class_mode="categorical", 
                subset="validation", 
                shuffle=False, 
                seed=30)

eval_val_generator=datagen2.flow_from_directory(
                data_dir,
                target_size=(img_width,img_height),
                batch_size=batch_size,
                classes= image_names,
                class_mode="categorical", 
                subset="validation", 
                shuffle=False, 
                seed=30) 



VGG16_classifier=VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(img_height, img_width, 3)))

for layer in VGG16_classifier.layers:
    layer.trainable=False

with tf.device(device):
  VGG16_model = tf.keras.Sequential([
                                     VGG16_classifier,
                                     MaxPooling2D(),
                                     Dense(img_height, activation='relu'),
                                     Flatten(),
                                     Dense(class_number, activation='softmax')
                                     ])
  
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


  VGG16_model.compile(
      optimizer='adam',
      loss=tf.losses.CategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
      )
  
  VGG16_model_history = VGG16_model.fit(
      train_generator,
      validation_data=val_generator,
      epochs=epochs,
      verbose=1,       
      callbacks=[callback])


train_loss, train_acc = VGG16_model.evaluate(train_generator)
print("\n Train Accuracy:", train_acc)
print("\n Train Loss:", train_loss)

test_loss, test_acc= VGG16_model.evaluate(val_generator)
print("\n Test Accuracy:", test_acc)
print("\n Test Loss:", test_loss)



# Save Model 
VGG16_model.save('VGG16_model')
VGG16_model.save('VGG16_model.h5')

VGG16_model_history_df = pd.DataFrame(VGG16_model_history.history) 
VGG16_model_history_df.to_csv('VGG16_model/history.csv')

tf.keras.utils.plot_model(
    VGG16_model, to_file='VGG16_model/architecture.png', show_shapes=False,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
)


Y_pred = VGG16_model.predict(val_generator, 1250 // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)

print('VGG16 Confusion Matrix- Validation')
conf_matrix = pd.DataFrame(
    confusion_matrix(val_generator.classes, y_pred), 
    index=['true: COVID_19_POSITIVE', 'true: COVID_19_NEGATIVE','true: VIRAL_PNEUMONIA'], 
    columns=['pred: COVID_19_POSITIVE', 'pred: COVID_19_NEGATIVE','pred: VIRAL_PNEUMONIA']
)
conf_matrix.to_csv("VGG16_model/confusion_matrix.jpeg")
print(conf_matrix)
print('\n')
print('Classification Report')
target_names = list(val_generator.class_indices.keys())
print(classification_report(val_generator.classes, y_pred, target_names=target_names))


Y_pred = VGG16_model.predict(train_generator, 2919 // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)

print('VGG16 Confusion Matrix for Training')
conf_matrix = pd.DataFrame(
    confusion_matrix(train_generator.classes, y_pred), 
    index=['true: COVID_19_POSITIVE', 'true: COVID_19_POSITIVE','true: VIRAL_PNEUMONIA'], 
    columns=['pred: COVID_19_POSITIVE', 'pred: COVID_19_NEGATIVE','pred: VIRAL_PNEUMONIA']
)
conf_matrix.to_csv("VGG16_model/confusion_matrix_train.jpeg")
print(conf_matrix)
print('\n')
print('Classification Report')
target_names = list(train_generator.class_indices.keys())
print(classification_report(train_generator.classes, y_pred, target_names=target_names))

import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
batch_size = 128
num_of_test_samples = 1250
predictions = VGG16_model.predict_generator(val_generator,  num_of_test_samples // batch_size+1)

y_pred = np.argmax(predictions, axis=1)

true_classes = val_generator.classes

class_labels = list(val_generator.class_indices.keys())   

print(class_labels)

print(confusion_matrix(val_generator.classes, y_pred))

report = classification_report(true_classes, y_pred, target_names=class_labels)
print(report)


plt.plot(VGG16_model_history.history['accuracy'])
plt.plot(VGG16_model_history.history['val_accuracy'])
plt.title('VGG16 Model Accuracy per Epoch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower center')
plt.show()

plt.plot(VGG16_model_history.history['loss'])
plt.plot(VGG16_model_history.history['val_loss'])
plt.title('VGG16 loss per Epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower center')
plt.show()

image_path = './COVID_19_NEGATIVE/Normal-12.png'
image = tf.keras.preprocessing.image.load_img(image_path,target_size=(224, 224))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = VGG16_model.predict(input_arr)
np.around(predictions, decimals=3)

pd.DataFrame(predictions).to_csv("VGG16_model/Predictions/normal1.csv")

#NORMAL
image_path = './COVID_19_NEGATIVE/Normal-1.png'
image = tf.keras.preprocessing.image.load_img(image_path,target_size=(224, 224))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = VGG16_model.predict(input_arr)
np.around(predictions, decimals=3)

pd.DataFrame(predictions).to_csv("VGG16_model/Predictions/normal2.csv")
image_path = './COVID_19_NEGATIVE/Normal-2138.png'
image = tf.keras.preprocessing.image.load_img(image_path,target_size=(224, 224))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = VGG16_model.predict(input_arr)
np.around(predictions, decimals=3)

pd.DataFrame(predictions).to_csv("VGG16_model/Predictions/normal3.csv")

# Begin Predictions and save to CSV

#COVID

image_path = './COVID_19_POSITIVE/COVID-34.png'
image = tf.keras.preprocessing.image.load_img(image_path,target_size=(224, 224))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = VGG16_model.predict(input_arr)
np.around(predictions, decimals=3)

pd.DataFrame(predictions).to_csv("VGG16_model/Predictions/covid1.csv")

#COVID

image_path = './COVID_19_POSITIVE/COVID-2635.png'
image = tf.keras.preprocessing.image.load_img(image_path,target_size=(224, 224))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = VGG16_model.predict(input_arr)
np.around(predictions, decimals=1)

pd.DataFrame(predictions).to_csv("VGG16_model/Predictions/covid2.csv")

#COVID

image_path = './COVID_19_POSITIVE/COVID-1000.png'
image = tf.keras.preprocessing.image.load_img(image_path,target_size=(224, 224))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = VGG16_model.predict(input_arr)
np.around(predictions, decimals=3)

pd.DataFrame(predictions).to_csv("VGG16_model/Predictions/covid3.csv")

#COVID

image_path = './COVID_19_POSITIVE/COVID-1566.png.png'
image = tf.keras.preprocessing.image.load_img(image_path,target_size=(224, 224))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = VGG16_model.predict(input_arr)
np.around(predictions, decimals=3)


pd.DataFrame(predictions).to_csv("VGG16_model/Predictions/covid4.csv")

#VIRAL_Pneumonia

image_path = './VIRAL_PNEUMONIA/Viral Pneumonia-2.png'
image = tf.keras.preprocessing.image.load_img(image_path,target_size=(224, 224))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = VGG16_model.predict(input_arr)
np.around(predictions, decimals=0)

pd.DataFrame(predictions).to_csv("VGG16_model/Predictions/vp1.csv")

#Viral_Pneumonia

image_path = './VIRAL_PNEUMONIA/Viral Pneumonia-300.png'
image = tf.keras.preprocessing.image.load_img(image_path,target_size=(224, 224))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = VGG16_model.predict(input_arr)
np.around(predictions, decimals=3)


pd.DataFrame(predictions).to_csv("VGG16_model/Predictions/vp2.csv")

#Viral_Pneumonia

image_path = './VIRAL_PNEUMONIA/Viral Pneumonia-699.png'
image = tf.keras.preprocessing.image.load_img(image_path,target_size=(224, 224))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = VGG16_model.predict(input_arr)
np.around(predictions, decimals=3)
pd.DataFrame(predictions).to_csv("VGG16_model/Predictions/vp3.csv")



df=pd.DataFrame(columns=['col1', 'col2', 'col3'])


# Verify columns to assigned names

for i in range(1,501):
  image_path = f'./PROJECT_DATASET/VIRAL_PNEUMONIA ({i+500}).png' 
  # image_path = f'Database/COVID19/COVID-19 ({i+400}).png'
  # image_path = f'Database/NORMAL/NORMAL ({i}).png' 
  image = tf.keras.preprocessing.image.load_img(image_path,target_size=(224, 224))
  input_arr = tf.keras.preprocessing.image.img_to_array(image)
  input_arr = np.array([input_arr])  # Convert single image to a batch.
  predictions = VGG16_model.predict(input_arr)
  preds=np.around(predictions, decimals=3)[0]
  df.loc[i-1] = preds
  print(i)

m = np.zeros_like(df.values)
m[np.arange(len(df)), df.values.argmax(1)] = 1

df1 = pd.DataFrame(m, columns = df.columns).astype(int)
df1.head()

df1[df1["col3"]==1].value_counts()
df1.head(50)

# PLOT ACCURACY
plt.plot(VGG16_model_history.history['accuracy'])
plt.plot(VGG16_model_history.history['val_accuracy'])
plt.title('VGG16 Model Accuracy per Epoch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower center')
plt.show()


# PLOT LOSS
plt.plot(VGG16_model_history.history['loss'])
plt.plot(VGG16_model_history.history['val_loss'])
plt.title('VGG16 loss per Epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower center')
plt.show()



