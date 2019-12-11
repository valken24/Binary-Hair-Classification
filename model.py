from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


def create_cnn(input_shape, n_classes, loss, optimizer, summary = True):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(1024, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5)) # regularización Dropout.
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=['acc','mse'])
    if summary:
        model.summary()

    return model



