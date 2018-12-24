import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from load import loader

training_data, val_data, test_data = loader('newdataset.pickle')

x_train, y_train = zip(*training_data)
x_val, y_val = zip(*val_data)
x_test, y_test = zip(*test_data)

#converting x to numpy array
x_train = np.asarray(x_train)
x_val = np.asarray(x_val)
x_test = np.asarray(x_test)
#check the dimensions of the dataset

print(np.shape(x_train))#2633, 96, 173
print(np.shape(x_test))#330, 96, 173
print(np.shape(y_train))#2633,

#checks how many families we have (10 from 0 to 10 exluding 9 (doesnt exist in this dataset))

list_of_names = []
for y in y_train:
    for name in list_of_names:
        if (name == y):
            break
    else:
        list_of_names.append(y)
    continue

print("number of names: " + str(len(list_of_names)))
#the cnn code

batch_size = 1024
num_classes = len(list_of_names) + 1 #9 is missing so we add 1
epochs = 200

# input image dimensions
img_rows, img_cols = np.shape(x_train)[1], np.shape(x_train)[2]

x_train = x_train.reshape(np.shape(x_train)[0],img_rows, img_cols, 1)
x_val = x_val.reshape(np.shape(x_val)[0], img_rows, img_cols, 1)
x_test = x_test.reshape(np.shape(x_test)[0],img_rows, img_cols, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# model = Sequential()
# model.add(Conv2D(128, kernel_size=(10, img_cols),
#                  activation='relu',
#                  input_shape=(img_rows,img_cols,1)))
# model.add(MaxPooling2D(pool_size=(5,1)))
# model.add(Flatten())
# model.add(Dense(11, activation='softmax'))
#
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.adam(),
#               metrics=['accuracy'])
# keras.utils.print_summary(model)
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_val, y_val))

model = keras.models.load_model('simple_model.h5')

score = model.evaluate(x_train, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('simple_model.h5')  # creates a HDF5 file 'my_model.h5'
