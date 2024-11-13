import os
import random
import cv2
from osgeo import gdal
import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def color_dict(labelFolder, classNum):
    colorDict = []
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath).astype(np.uint32)
        if (len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        img_new = img[:, :, 0] * 1000000 + img[:, :, 1] * 1000 + img[:, :, 2]
        unique = np.unique(img_new)
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        colorDict = sorted(set(colorDict))
        if (len(colorDict) == classNum):
            break
    colorDict_RGB = []
    for k in range(len(colorDict)):
        color = str(colorDict[k]).rjust(9, '0')
        color_RGB = [int(color[0: 3]), int(color[3: 6]), int(color[6: 9])]
        colorDict_RGB.append(color_RGB)
    colorDict_RGB = np.array(colorDict_RGB)
    colorDict_GRAY = colorDict_RGB.reshape((colorDict_RGB.shape[0], 1, colorDict_RGB.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_RGB, colorDict_GRAY


def readTif(fileName):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
    return GdalImg_data


def dataPreprocess(img, label, classNum, colorDict_GRAY):
    img = img / 255.0
    for i in range(colorDict_GRAY.shape[0]):
        label[label == colorDict_GRAY[i][0]] = i
    new_label = np.zeros(label.shape + (classNum,))
    for i in range(classNum):
        new_label[label == i, i] = 1
    label = new_label
    return (img, label)


def trainGenerator(batch_size, train_image_path, train_label_path, classNum, colorDict_GRAY,
                   resize_shape=(256, 256, 3)):
    imageList = os.listdir(train_image_path)
    labelList = os.listdir(train_label_path)
    img = readTif(train_image_path + "/" + imageList[0])
    img = img.swapaxes(1, 0)
    img = img.swapaxes(1, 2)
    while (True):
        img_generator = np.zeros((batch_size, img.shape[0], img.shape[1], img.shape[2]), np.uint8)
        label_generator = np.zeros((batch_size, img.shape[0], img.shape[1]), np.uint8)
        if (resize_shape != None):
            img_generator = np.zeros((batch_size, resize_shape[0], resize_shape[1], resize_shape[2]), np.uint8)
            label_generator = np.zeros((batch_size, resize_shape[0], resize_shape[1]), np.uint8)


        rand = random.randint(0, len(imageList) - batch_size)

        for j in range(batch_size):
            img = readTif(train_image_path + "/" + imageList[rand + j])
            img = img.swapaxes(1, 0)
            img = img.swapaxes(1, 2)
            if (resize_shape != None):
                img = np.resize(img, (resize_shape[0], resize_shape[1], 3))
            img_generator[j] = img

            label = readTif(train_label_path + "/" + labelList[rand + j]).astype(np.uint8)
            if (len(label.shape) == 3):
                label = label.swapaxes(1, 0)
                label = label.swapaxes(1, 2)
                label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
            if (resize_shape != None):
                label = cv2.resize(label, (resize_shape[0], resize_shape[1]))
            label_generator[j] = label
        img_generator, label_generator = dataPreprocess(img_generator, label_generator, classNum, colorDict_GRAY)

        yield (img_generator, label_generator)


def testGenerator(test_iamge_path, resize_shape=None):
    imageList = os.listdir(test_iamge_path)
    for i in range(len(imageList)):
        img = readTif(test_iamge_path + "/" + imageList[i])
        img = img.swapaxes(1, 0)
        img = img.swapaxes(1, 2)
        img = img / 255.0
        if (resize_shape != None):
            img = cv2.resize(img, (resize_shape[0], resize_shape[1]))
        img = np.reshape(img, (1,) + img.shape)
        yield img


def saveResult(test_image_path, test_predict_path, model_predict, color_dict, output_size):
    imageList = os.listdir(test_image_path)
    for i, img in enumerate(model_predict):
        channel_max = np.argmax(img, axis=-1)
        img_out = np.uint8(color_dict[channel_max.astype(np.uint8)])
        img_out = cv2.resize(img_out, (output_size[0], output_size[1]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(test_predict_path + "/" + imageList[i][:-4] + ".png", img_out)
