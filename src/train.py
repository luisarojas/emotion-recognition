from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from model import create_model
from math import ceil

if __name__ == "__main__":
    print("---------Training---------")
    model = create_model()
    model.compile(loss="categorical_crossentropy",
                  optimizer="rmsprop",
                  metric=["accuracy"])

    
