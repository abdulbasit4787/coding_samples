from modeling import unet
from data_process import testGenerator, saveResult, color_dict
import os

model_path = "/home/nick/PycharmProjects/project1/dataset/Model/ep150-loss1.101-val_loss1.153.hdf5"
test_image_path = "/home/nick/PycharmProjects/project1/dataset/png/test"
save_path = "/home/nick/PycharmProjects/project1/dataset/png/predict"

test_num = len(os.listdir(test_image_path))
classNum = 2
input_size = (512, 512, 3)
output_size = (1500, 1500)

train_label_path = "/home/nick/PycharmProjects/project1/dataset/png/train_labels"

colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)

model = unet(model_path)

testGene = testGenerator(test_image_path, input_size)
results = model.predict(testGene,test_num, verbose = 1)

saveResult(test_image_path, save_path, results, colorDict_GRAY, output_size)



