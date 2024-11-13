import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modeling import unet
from callbacks import ExponentDecayScheduler, LossHistory
from data_process import trainGenerator, color_dict, testGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
import datetime
import xlwt
import os
from tensorflow.keras.optimizers import Adam
import pandas as pd
import torch
import pandas as pd
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torch_snippets import *
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torchvision.models import vgg16_bn
import tensorflow as tf
import torch

IMG_ROOT = '/home/nick/PycharmProjects/project1.2/dataset/'
df = pd.read_csv('/home/nick/PycharmProjects/project1.2/dataset/metadata.csv')
train_image_path = r"/home/nick/PycharmProjects/project1/dataset/png/train"
train_label_path = r"/home/nick/PycharmProjects/project1/dataset/png/train_labels"
validation_image_path = r"/home/nick/PycharmProjects/project1/dataset/png/val"
validation_label_path = r"/home/nick/PycharmProjects/project1/dataset/png/val_labels"

train_df = df.loc[df['split'] == 'train']
tst_df = df.loc[df['split'] == 'test']
print("Training dataset size:", len(train_df))
print("Test dataset size:", len(tst_df))

tfms = transforms.Compose([
#     transforms.RandomHorizontalFlip(0.3),
#     transforms.RandomRotation(40),
#     transforms.RandomVerticalFlip(0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], # imagenet
                         [0.229, 0.224, 0.225])])
batch_size = 2
classNum = 3
imageList = os.listdir(train_image_path)
labelList = os.listdir(train_label_path)
train_num = len(os.listdir(train_image_path))
validation_num = len(os.listdir(validation_image_path))
steps_per_epoch = train_num / batch_size
validation_steps = validation_num / batch_size
colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)
print(imageList)





