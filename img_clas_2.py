import tensorflow as tf   

# Display the version
print(tf.__version__)     

# other imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

# Load in the data
cifar10 = tf.keras.datasets.cifar10 # location of the data set through api 

# Distribute it to train and test set

(x_train, y_train), (x_test, y_test) = cifar10.load_data() # desterbute data from cifar 10 to 4 different list 

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # print shape of each file 

# Reduce pixel values

x_train, x_test = x_train / 255.0, x_test / 255.0 
# picture in shape 1-256 so we want to have a 0 - 1 . 
# This enables our model to easily track trends and efficient training

# flatten the label values
y_train, y_test = y_train.flatten(), y_test.flatten()

# flatten(in simple words rearrange them in form of a row) the label values using the flatten() function. 

# visualize data by plotting images
fig, ax = plt.subplots(5, 5)
k = 0

for i in range(5):
    for j in range(5):
        ax[i][j].imshow(x_train[k], aspect='auto')
        k += 1

plt.show()

# number of classes
K = len(set(y_train))

# calculate total number of classes 
# for output layer
print("number of classes:", K)

# Build the model using the functional API
# input layer
i = Input(shape=x_train[0].shape)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)

x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)

x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)

x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.2)(x)

# Hidden layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)

# last hidden layer i.e.. output layer
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

# model description
model.summary()

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit
r = model.fit(
  x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# Fit with data augmentation
# Note: if you run this AFTER calling
# the previous model.fit()
# it will CONTINUE training where it left off
batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
  width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

train_generator = data_generator.flow(x_train, y_train, batch_size)
steps_per_epoch = x_train.shape[0] // batch_size

r = model.fit(train_generator, validation_data=(x_test, y_test),
              steps_per_epoch=steps_per_epoch, epochs=50)

# Compile 

model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

# Fit 

r = model.fit(
    x_train,y_train,validation_data = (x_test , y_test), epochs = 50
)

# Fit with data augmentation 
# None: if you run this After Calling
# the previous model.fit()
# it will Continue training where it left off 

batch_size = 32 
data_generation = tf.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range = 0.1 , height_shift_range = 0.1 , horizontal_flip = True)
train_generator = data_generator.flow(x_train,y_train,batch_size)
steps_per_epoch = X_train.shape[0]//batch_size

r = model.fit(train_generator,validation_data = (x_test,y_test),steps_per_epoch = steps_per_epoch , epoch = 50)

# PLot accuracy per iteration 

plt.plot(r.history['accuracy'],label = 'acc',color = 'red')
plt.plot(r.history['val_accuracy'],label = 'val_acc',color = 'green')
plt.legend()

Labels = '''airplane automobile birds cat deerdog frog horship truck'''.split()

# select the image from our tast dataset

image_number = 0 

# display the image 
plt.imshow(x_test[image_number])

# load the image in an array 

n = np.array(x_test[image_number])

# reshape it 

p = n.reshape(1,32,32,3)

# pass in the network for prediction and save the predicted label 

predicted_label = labels[model.predict(p).argmax()]

# load the original label 

original_label = labels[y_test[image_number]]

# display the result 
print('Original label is {} and predicted label is {}'.format(
    original_label, predicted_label
))

# save the model 
model.save('geeksforgeeks.h5')
