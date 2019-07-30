import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow
from keras.models import  Sequential
from keras.layers import Conv2D, Flatten, Dense, Input, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import read_binary
import cv2


def get_data_and_labels(dataset):
    X = np.zeros((len(dataset), 28, 28))
    y = np.zeros((len(dataset), 10))
    for i in range(len(dataset)):
        X[i] = dataset[i][0]
        label = dataset[i][1]
        y[i][label] = 1
    return X, y


training_set = read_binary.get_data_from_pkl("train")
X_train, y_train = get_data_and_labels(training_set)

test_set = read_binary.get_data_from_pkl("test")
X_test, y_test = get_data_and_labels(test_set)


# index = 0
# for digit in training_set:
#     print digit
#     for i in range(len(digit)):
#         X[index] = digit[i]
#
# print X.shape

# digits = load_digits()
# X = digits.images
# X = np.expand_dims(X, axis=1)
# # X = np.ones((digits.images.shape[0], digits.images.shape[1] * digits.images.shape[2]), dtype=np.float32)
# # for i in range(digits.images.shape[0]):
# #     index = 0
# #     for row in range(digits.images.shape[1]):
# #         for col in range(digits.images.shape[2]):
# #             X[i][index] = float(digits.images[i][row][col])
# #             index += 1
# targets = digits.target
# y = np.zeros((targets.shape[0], 10), np.uint8)
#
# for index in range(len(targets)):
#     y[index][targets[index]] = 1


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# # X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_train = np.expand_dims(X_train, axis=1)


input_shape = (1, 28, 28, )
model = Sequential()
# input = Input(shape=input_shape)
# model.add(Input(shape=input_shape))
model.add(Conv2D(20, (2, 2), data_format='channels_first', input_shape=input_shape))
model.add(Conv2D(20, (4, 4), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(40, (5, 5), activation='relu'))
model.add(MaxPooling2D((3, 3)))
model.add(Flatten())
model.add(Dense(150, activation='relu'))
# model.add(Dense(10, ac))
# model.add(Conv2D(20, (2, 2), data_format='channels_first', input_shape=input_shape, activation='relu'))
# model.add(Conv2D(10, (2, 2), activation='relu'))
# model.add(Conv2D(5, (2, 2), activation='relu'))
# model.add(Flatten())
# model.add(Dense(10, activation='relu', input_shape=(64,)))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
filepath = "weights-improvement-digits-conv2d-2907.hdf5"
model.load_weights(filepath)

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=500, callbacks=callbacks_list, verbose=0)
# score = model.evaluate(X_test, y_test)
# model.save_weights('model-conv2d-2907.h5')
# print 'score: ', score[1]
X_test = np.expand_dims(X_test, axis=1)
preds = model.predict(X_test)
count = 0
for index in range(len(preds)):
    predict = np.argmax(preds[index])
    # print predict, y_test[index]
    count += 1 if y_test[index, predict] == 1 else 0
print count*1./len(y_test)