import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD,Adam
from keras.models import load_model
import numpy as np
import os
import argparse
import sys
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'
num_classes = 10
# Import data (Samples, 28, 28, 1)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train /= 255
x_test /= 255
#yTrain = np.squeeze(yTrain)
#yTest = np.squeeze(yTest)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Dense(input_dim=3072,units=128,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=10, activation='softmax'))
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#optr=keras.optimizers.SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
#sgd = SGD(lr=0.01, decay=1e-6)
adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])



if __name__ == '__main__':
    args = sys.argv
    try:
        if len(args) == 2 and args[1] == 'train':
            # train the data
            hist = model.fit(x_train, y_train, epochs=15, batch_size=100, validation_data=(x_test, y_test))

            print('       training loss  training acc%   testing loss   testing acc%')
            for i in range(0, 14):
                print('epoch{}:'.format(i + 1), '%.4f' % hist.history['loss'][i], '        ',
                      '%.4f' % hist.history['acc'][i], '        ', '%.4f' % hist.history['val_loss'][i], '        ',
                      '%.4f' % hist.history['val_acc'][i])
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            model_path = os.path.join(save_dir, model_name)
            model.save(model_path)

            # print('Saved trained model at %s ' % model_path)
        elif len(args) == 2 and args[1] == 'test':
            # read the trained model
            model_path = os.path.join(save_dir, model_name)
            model = load_model(model_path)

            # Score trained model.
            scores = model.evaluate(x_test, y_test)
            print('Test loss:', scores[0])
            print('Test accuracy:', scores[1])
        else:
            print('Usage:')
            print('===> "python classify.py train"')
            print('===> "python classify.py test"')
    except Exception as ex:
        print('Usage:')
        print('===> "python classify.py train"')
        print('===> "python classify.py test"')
        print('ERROR:', ex)

