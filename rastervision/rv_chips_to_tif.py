import albumentations as A
import rasterio
import numpy as np
from PIL import Image
from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *
from rastervision.pipeline.config import register_config, Config, Field
import albumentations as A
import os
from matplotlib import pyplot as plt
from rastervision.core.data import *
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset
from pathlib import Path
from rastervision.pytorch_learner.dataset import discover_images
import torch
torch.cuda.empty_cache()
from rastervision.pytorch_learner import SemanticSegmentationImageDataset


train_image_uri = '/home/nicku/Downloads/EORSSD-dataset-master/train'
train_label_uri = '/home/nicku/Downloads/EORSSD-dataset-master/train_labels'
val_image_uri = '/home/nicku/Downloads/EORSSD-dataset-master/test'
val_label_uri = '/home/nicku/Downloads/EORSSD-dataset-master/test_labels'
test_img_uri = '/home/nicku/PycharmProjects/pythonProject/ma_dataset/pred.tif'

pred = '/home/nicku/PycharmProjects/pythonProject/results'
pred_bw = '/home/nicku/PycharmProjects/pythonProject/results'
pred_overlap = '/home/nicku/PycharmProjects/pythonProject/results'
pred_tif = '/home/nicku/PycharmProjects/pythonProject/results'
learner_dir = '/home/nicku/PycharmProjects/pythonProject/model'
saved_model = '/home/nicku/PycharmProjects/pythonProject/model/model-bundle.zip'


class_config = ClassConfig(
    names=['background', 'building'],
    colors=['black', 'white'],
    null_class='background'
)
class_config.ensure_null_class()

viz = SemanticSegmentationVisualizer(
    class_names=class_config.names, class_colors=class_config.colors)

data_augmentation_transform = A.Compose([
    A.Flip(),
    A.Resize(width=256, height=256),
    A.ShiftScaleRotate(),
    A.OneOf([
        A.HueSaturationValue(hue_shift_limit=10),
        A.RGBShift(),
        A.ToGray(),
        A.ToSepia(),
        A.RandomBrightness(),
        A.RandomGamma(),
    ]),
    A.CoarseDropout(max_height=32, max_width=32, max_holes=5)
])

train_ds = SemanticSegmentationImageDataset(
    img_dir=train_image_uri,
    label_dir=train_label_uri,
    transform=data_augmentation_transform,

)

val_ds = SemanticSegmentationImageDataset(
    img_dir=val_image_uri,
    label_dir=val_label_uri,
    transform=A.Resize(256, 256)
)

data_cfg = SemanticSegmentationGeoDataConfig(
    img_channels=3,
    class_names=class_config.names,
    class_colors=class_config.colors,
    num_workers=0,  # increase to use multi-processing
)

solver_cfg = SolverConfig(
    batch_sz=2,
    lr=6e-5,
    class_loss_weights=[1., 10.],
)

model1 = SemanticSegmentationModelConfig(backbone=Backbone.resnet101, pretrained=True, init_weights=None)

model2 = torch.hub.load(
    'AdeelH/pytorch-fpn:0.3',
    'make_fpn_resnet',
    name='resnet18',
    fpn_type='panoptic',
    num_classes=len(class_config),
    fpn_channels=128,
    in_channels=3,
    out_size=(256, 256),
    pretrained=True)

learner_cfg = SemanticSegmentationLearnerConfig(data=data_cfg, solver=solver_cfg)#, model=model1)

learner = SemanticSegmentationLearner(
    cfg=learner_cfg,
    output_dir=learner_dir,
    model = model2,
    train_ds=train_ds,
    valid_ds=val_ds,
)
learner.train(epochs=3)
learner.save_model_bundle()

class customSemanticSegmentationDataReader(Dataset):

    def __init__(self, img_dir: str):

        self.img_dir = Path(img_dir)
        img_paths = discover_images(img_dir)
        self.img_paths = sorted(img_paths, key=lambda p: p.stem)


    def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.img_paths[ind]
        x = load_image(img_path)
        return x

    def __len__(self):
        return len(self.img_paths)


class customSemanticSegmentationImageDataset(ImageDataset):
    def __init__(self, img_dir: str, *args, **kwargs):

        ds = customSemanticSegmentationDataReader(img_dir)
        super().__init__(
            ds,
            *args,
            **kwargs,
            transform_type=TransformType.semantic_segmentation)


"""ds = customSemanticSegmentationImageDataset(
    img_dir=test_img_uri,
    transform=A.Resize(256, 256)
    )"""
ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris( #if your test dataset have .tif
    class_config=class_config,
    image_uri=test_img_uri,
    size=256,
    stride=256,
    transform=A.Resize(256, 256))
predictions = learner.predict_dataset(
    ds,
    raw_out=True,
    numpy_out=True,
    predict_kw=dict(out_shape=(256, 256)),
    progress_bar=True)

pred_labels = SemanticSegmentationLabels.from_predictions(
    ds.windows,
    predictions,
    smooth=True,
    extent=ds.scene.extent,
    num_classes=len(class_config))

scores = pred_labels.get_score_arr(pred_labels.extent)
scores_building = scores[0]
scores_background = scores[1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout(w_pad=-2)
ax1.imshow(scores_building, cmap='plasma')
ax1.axis('off')
ax1.set_title('building')
fig.savefig('/home/nicku/PycharmProjects/pythonProject/results/building.png')
ax2.imshow(scores_background, cmap='plasma')
ax2.axis('off')
ax2.set_title('background')
fig.savefig('/home/nicku/PycharmProjects/pythonProject/results/background.png')
plt.show()

pred_labels.save(
    uri=pred_tif,
    crs_transformer=ds.scene.raster_source.crs_transformer,
    class_config=class_config)