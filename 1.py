import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import segmentation_models_pytorch.utils as utils
from torch.utils.data import DataLoader
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os, cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
torch.cuda.empty_cache()
import tensorflow as tf


def check_gpu_usage():
    tf_gpu_available = tf.test.is_gpu_available()
    torch_gpu_available = torch.cuda.is_available()

    if tf_gpu_available:
        print("TensorFlow is using GPU.")
    else:
        print("TensorFlow is using CPU.")

    if torch_gpu_available:
        print("PyTorch is using GPU.")
    else:
        print("PyTorch is using CPU.")

if __name__ == "__main__":
    check_gpu_usage()

Train_dir = 'C:/Users/DELL/Downloads/Compressed/Building_Dataset-main/Building_Dataset-main/train/'
x_train_dir = os.path.join(Train_dir, 'image')
y_train_dir = os.path.join(Train_dir, 'mask')
Val_dir = 'C:/Users/DELL/Downloads/Compressed/Building_Dataset-main/Building_Dataset-main/val/'
x_valid_dir = os.path.join(Val_dir, 'image')
y_valid_dir = os.path.join(Val_dir, 'mask')

"""Train_dir = 'C:/Users/DELL/Downloads/Compressed/instance-segmentation-building-dataset-of-china/building/train/'
x_train_dir = os.path.join(Train_dir, 'Images_png')
y_train_dir = os.path.join(Train_dir, 'PNG')
Val_dir = 'C:/Users/DELL/Downloads/Compressed/instance-segmentation-building-dataset-of-china/building/test/'
x_valid_dir = os.path.join(Val_dir, 'Images_png')
y_valid_dir = os.path.join(Val_dir, 'PNG')
"""
test_dir = "C:/Users/DELL/PycharmProjects/building_footprints/new_dataset - Copy/taizhou_dataset/"
x_test_dir = os.path.join(test_dir, 'test')
#y_test_dir = os.path.join(DATA_DIR, 'masks')
class_dict = pd.read_csv("C:/Users/DELL/PycharmProjects/autonomous/raw datasets/ma_dataset/label_class_dict.csv")
class_names = class_dict['name'].tolist()
class_rgb_values = class_dict[['r','g','b']].values.tolist()

#print('All dataset classes and their corresponding RGB values in labels:')
#print('Class Names: ', class_names)
#print('Class RGB values: ', class_rgb_values)
select_classes = ['background', 'building']

select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def reverse_one_hot(image):
    x = np.argmax(image, axis=-1)
    return x


def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x


class BuildingsDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            images_dir,
            masks_dir,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
    ):

        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.image_paths)


dataset = BuildingsDataset(x_train_dir, y_train_dir, class_rgb_values=select_class_rgb_values)
random_idx = random.randint(0, len(dataset)-1)
image, mask = dataset[2]


def get_training_augmentation():
    train_transform = [
        album.RandomCrop(height=256, width=256, always_apply=True),
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),
        album.ToFloat(max_value=255),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)



augmented_dataset = BuildingsDataset(
    x_train_dir, y_train_dir,
    augmentation=get_training_augmentation(),
    class_rgb_values=select_class_rgb_values,
)

random_idx = random.randint(0, len(augmented_dataset)-1)
# Different augmentations on a random image/mask pair (256*256 crop)


CLASSES = class_names

ENCODER = 'resnet152'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation #sigmoid
model_delete = TRAINING = false1 =  True
EPOCHS = 600
learning_rate = 1e-3
batch_size_train = 4
batch_size_val = 4
in_channels = 3

if model_delete:
    model_file = 'C:/Users/DELL/PycharmProjects/building_footprints/model/best_model1.pth'
    if os.path.exists(model_file):
        os.remove(model_file)

model = smp.UnetPlusPlus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
    in_channels=in_channels
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


train_dataset = BuildingsDataset(
    x_train_dir, y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=select_class_rgb_values,
)

valid_dataset = BuildingsDataset(
    x_valid_dir, y_valid_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=select_class_rgb_values,
)

train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)#, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size_val, shuffle=False)#, num_workers=1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5)]
optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr= learning_rate),
])

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1, T_mult=2, eta_min=5e-5,
)

# load best saved model checkpoint from previous commit (if present)
if os.path.exists('C:/Users/DELL/PycharmProjects/building_footprints/model/best_model1.pth'):
    model = torch.load('C:/Users/DELL/PycharmProjects/building_footprints/model/best_model1.pth', map_location=DEVICE)
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

if TRAINING:

    best_iou_score = 0.0
    best_accuracy = 0.0
    best_precision = 0.0
    train_logs_list, valid_logs_list = [], []

    for i in range(0, EPOCHS):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, 'C:/Users/DELL/PycharmProjects/building_footprints/model/best_model1.pth')
            print('Model saved!')

        """if best_accuracy < valid_logs['accuracy'] and best_precision < valid_logs['precision']:
            best_accuracy = valid_logs['accuracy']
            best_precision = valid_logs['precision']
            torch.save(model, 'C:/Users/DELL G3/PycharmProjects/autonomous/model/best_model1.pth')
            print('Model saved!')"""


if os.path.exists('C:/Users/DELL/PycharmProjects/building_footprints/model/best_model1.pth'):
    best_model1 = torch.load('C:/Users/DELL/PycharmProjects/building_footprints/model/best_model1.pth', map_location=DEVICE)
    print('Loaded UNet model from this run.')

elif os.path.exists('C:/Users/DELL/PycharmProjects/building_footprints/model/best_model1.pth'):
    best_model1 = torch.load('C:/Users/DELL/PycharmProjects/building_footprints/model/best_model1.pth', map_location=DEVICE)
    print('Loaded UNet model from a previous commit.')

class customBuildingsDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            images_dir,
            #masks_dir,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
    ):

        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        #self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        #mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        #mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image

    def __len__(self):
        return len(self.image_paths)

test_dataset = customBuildingsDataset(
    x_test_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=select_class_rgb_values,
)

test_dataloader = DataLoader(test_dataset)

test_dataset_vis = customBuildingsDataset(
    x_test_dir,
    augmentation=get_validation_augmentation(),
    class_rgb_values=select_class_rgb_values,
)

random_idx = random.randint(0, len(test_dataset_vis)-1)
image = test_dataset_vis[random_idx]




def crop_image(image, target_image_dims=[256, 256, 3]):
    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[
           padding:image_size - padding,
           :,
           ]


sample_preds_folder = 'C:/Users/DELL/PycharmProjects/building_footprints/new_dataset - Copy/predict1/'
if not os.path.exists(sample_preds_folder):
    os.makedirs(sample_preds_folder)

for idx in range(len(test_dataset)):
    image = test_dataset[idx]
    image_vis = crop_image(test_dataset_vis[idx][0].astype('uint8'))
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    # Predict test image
    pred_mask = best_model1(x_tensor)
    pred_mask = pred_mask.detach().squeeze().cpu().numpy()
    # Convert pred_mask from `CHW` format to `HWC` format
    pred_mask = np.transpose(pred_mask, (1, 2, 0))
    # Get prediction channel corresponding to building
    pred_building_heatmap = pred_mask[:, :, select_classes.index('building')]
    pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values))
    # Convert gt_mask from `CHW` format to `HWC` format
    #gt_mask = np.transpose(gt_mask, (1, 2, 0))
    #gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values))

    output_path = os.path.join(sample_preds_folder, os.path.basename(test_dataset.image_paths[idx]))# f"sample_pred_{idx}.png")
    cv2.imwrite(output_path, pred_mask)

    #cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"),np.hstack([image_vis, gt_mask, pred_mask])[:, :, ::-1])


test_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)
"""
valid_logs = test_epoch.run(test_dataloader)
print("Evaluation on Test Data: ")
print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")

if false1:
    train_logs_df = pd.DataFrame(train_logs_list)
    valid_logs_df = pd.DataFrame(valid_logs_list)
    train_logs_df.T

    plt.figure(figsize=(20, 8))
    plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist(), lw=3, label='Train')
    plt.plot(valid_logs_df.index.tolist(), valid_logs_df.iou_score.tolist(), lw=3, label='Valid')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('IoU Score', fontsize=20)
    plt.title('IoU Score Plot', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('C:/Users/DELL/PycharmProjects/building_footprints/new_dataset - Copy/iou_score_plot.png')
    # plt.show()

    plt.figure(figsize=(20, 8))
    plt.plot(train_logs_df.index.tolist(), train_logs_df.dice_loss.tolist(), lw=3, label='Train')
    plt.plot(valid_logs_df.index.tolist(), valid_logs_df.dice_loss.tolist(), lw=3, label='Valid')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Dice Loss', fontsize=20)
    plt.title('Dice Loss Plot', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('C:/Users/DELL/PycharmProjects/building_footprints/new_dataset - Copy/dice_loss_plot.png')
    # plt.show()"""