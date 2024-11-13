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
#from tensorflow.keras.optimizers import Adam
from losses import  tversky_loss, Focal_Loss, focal_tversky
import segmentation_models_pytorch as smp
import torch
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


train_image_path = r"/home/nick/PycharmProjects/project1/dataset/png/train"
train_label_path = r"/home/nick/PycharmProjects/project1/dataset/png/train_labels"
validation_image_path = r"/home/nick/PycharmProjects/project1/dataset/png/val"
validation_label_path = r"/home/nick/PycharmProjects/project1/dataset/png/val_labels"

batch_size = 2
classNum = 2
input_size = (512, 512, 3)
epochs = 3
learning_rate = 1e-4 #1e-5
pretrained_weights = premodel_path = None
#model_path = "Data/5_18/model_save"

train_num = len(os.listdir(train_image_path))
validation_num = len(os.listdir(validation_image_path))
steps_per_epoch = train_num / batch_size
validation_steps = validation_num / batch_size
colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)





train_Generator = trainGenerator(batch_size,
                                 train_image_path,
                                 train_label_path,
                                 classNum ,
                                 colorDict_GRAY,
                                 input_size)

validation_data = trainGenerator(batch_size,
                                 validation_image_path,
                                 validation_label_path,
                                 classNum,
                                 colorDict_GRAY,
                                 input_size)

model = unet(input_size = input_size, classNum = classNum)
#model.summary()



logging         = TensorBoard(log_dir = '/home/nick/PycharmProjects/project1/logs/', write_graph = True)
#reduce_lr = ExponentDecayScheduler(decay_rate=0.96, verbose=1)
early_stopping  = EarlyStopping(monitor = 'loss' , patience = 10, verbose=1, mode = "min")
loss_history    = LossHistory('/home/nick/PycharmProjects/project1/logs/')


checkpoint = ModelCheckpoint(
    filepath = '/home/nick/PycharmProjects/project1/model/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.hdf5',
    monitor = 'loss',
    verbose = 1,
    save_best_only = False,
    mode = 'min',
) #save_weights_only = False, save_freq='epoch'

cls_weights = np.array([1,3], np.float32)

model.compile(optimizer='adam', loss=Focal_Loss(cls_weights), metrics=['accuracy'])
#model.compile(optimizer='adam', loss="categorical_crossentropy",metrics=['accuracy'])  # optimizer='adam'
#model.compile(optimizer='adam', loss=focal_loss, metrics=['accuracy'])  # optimizer='adam'
#model.compile(optimizer='adam', loss=tversky_loss, metrics=['accuracy'])
#model.compile(optimizer='adam', loss=tversky, metrics=['accuracy'])
#model.compile(optimizer='adam', loss=focal_tversky, metrics=['accuracy'])
#model.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

#model.compile(optimizer=Adam(learning_rate = learning_rate), loss=Focal_Loss(cls_weights), metrics=['accuracy'])
#model.compile(optimizer=Adam(learning_rate = learning_rate), loss="categorical_crossentropy",metrics=['accuracy'])  # optimizer='adam'
#model.compile(optimizer=Adam(learning_rate=learning_rate), loss=focal_loss, metrics=['accuracy'])  # optimizer='adam'
#model.compile(optimizer=Adam(learning_rate=learning_rate), loss=tversky_loss, metrics=['accuracy'])
#model.compile(optimizer=Adam(learning_rate=learning_rate), loss=tversky, metrics=['accuracy'])
#model.compile(optimizer=Adam(learning_rate=learning_rate), loss=focal_tversky, metrics=['accuracy'])
#model.compile(optimizer=Adam(learning_rate=learning_rate), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

# model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate), loss=Focal_Loss(cls_weights), metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate), loss="categorical_crossentropy",metrics=['accuracy'])  # optimizer='adam'
# model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate), loss=focal_loss, metrics=['accuracy'])  # optimizer='adam'
# model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate), loss=tversky_loss, metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate), loss=tversky, metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate), loss=focal_tversky, metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

start_time = datetime.datetime.now()

print(train_Generator)
print(validation_data)
history = model.fit(train_Generator,
                    steps_per_epoch = steps_per_epoch,
                    epochs = epochs,
                    callbacks=[early_stopping, logging, checkpoint, loss_history],
                    validation_data=validation_data,
                    validation_steps=validation_steps,
                    verbose = 1)

end_time = datetime.datetime.now()
log_time = "Total training time: " + str((end_time - start_time).seconds / 60) + "m"
time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig("accuracy_%s.png"%time, dpi = 300)
plt.figure()
plt.plot(epochs, loss, 'r', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("loss_%s.png"%time, dpi = 300)
plt.show()