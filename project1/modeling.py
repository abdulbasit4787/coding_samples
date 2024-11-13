from tensorflow.keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
import numpy as np
from keras.optimizers import Adam
from losses import *
import torch
import pandas as pd
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def unet(input_size=(512, 512, 3), classNum=2):
    inputs = Input(input_size)
    conv1 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs))
    conv1 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1))
    conv2 = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = BatchNormalization()(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2))
    conv3 = BatchNormalization()(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = BatchNormalization()(
        Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3))
    conv4 = BatchNormalization()(
        Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4))
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = BatchNormalization()(
        Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4))
    conv5 = BatchNormalization()(
        Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5))
    drop5 = Dropout(0.5)(conv5)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))

    try:
        merge6 = concatenate([drop4, up6], axis=3)
    except:
        merge6 = concatenate([drop4, up6], mode='concat', concat_axis=3)
    conv6 = BatchNormalization()(
        Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6))
    conv6 = BatchNormalization()(
        Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6))

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    try:
        merge7 = concatenate([conv3, up7], axis=3)
    except:
        merge7 = concatenate([conv3, up7], mode='concat', concat_axis=3)
    conv7 = BatchNormalization()(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7))
    conv7 = BatchNormalization()(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7))

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    try:
        merge8 = concatenate([conv2, up8], axis=3)
    except:
        merge8 = concatenate([conv2, up8], mode='concat', concat_axis=3)
    conv8 = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8))
    conv8 = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8))

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    try:
        merge9 = concatenate([conv1, up9], axis=3)
    except:
        merge9 = concatenate([conv1, up9], mode='concat', concat_axis=3)
    conv9 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9))
    conv9 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9))
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(classNum, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model
# ---------------------------------------------------------------------

