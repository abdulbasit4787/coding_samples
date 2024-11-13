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
import torch
torch.cuda.empty_cache()
from rastervision.core.evaluation import *
from torch.utils.data import Dataset
from pathlib import Path
from rastervision.pytorch_learner.dataset import discover_images


train_image_uri = '/home/nicku/PycharmProjects/pythonProject/ma_dataset/test.tif'
train_label_uri = '/home/nicku/PycharmProjects/pythonProject/ma_dataset/test.geojson'
val_image_uri = '/home/nicku/PycharmProjects/pythonProject/ma_dataset/valid.tif'
val_label_uri = '/home/nicku/PycharmProjects/pythonProject/ma_dataset/valid.geojson'
test_img_uri = "/home/nicku/PycharmProjects/pythonProject/ma_dataset/TJ"

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

train_ds = SemanticSegmentationRandomWindowGeoDataset.from_uris(
    class_config=class_config,
    image_uri=train_image_uri,
    label_vector_uri=train_label_uri,
    label_vector_default_class_id=class_config.get_class_id('building'),
    size_lims=(150, 200),
    out_size=256,
    max_windows=400,
    transform=data_augmentation_transform)


val_ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
    class_config=class_config,
    image_uri=val_image_uri,
    label_vector_uri=val_label_uri,
    label_vector_default_class_id=class_config.get_class_id('building'),
    size=256,
    stride=200,
    transform=A.Resize(256, 256))


data_cfg = SemanticSegmentationGeoDataConfig(
    img_channels=3,
    class_names=class_config.names,
    class_colors=class_config.colors,
    num_workers=2,  # increase to use multi-processing
)
solver_cfg = SolverConfig(
    batch_sz=4,
    lr=6e-5,
    class_loss_weights=[1., 10.]#[0.5, 0.5] #

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

learner.train(epochs=1)
learner.save_model_bundle()

pred_ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris( #if your test dataset have .tif
    class_config=class_config,
    image_uri=test_img_uri,
    size=256,
    stride=256,
    transform=A.Resize(256, 256))
predictions = learner.predict_dataset(
    pred_ds,
    raw_out=True,
    numpy_out=True,
    predict_kw=dict(out_shape=(256, 256)),
    progress_bar=True)

pred_labels = SemanticSegmentationLabels.from_predictions(
    pred_ds.windows,
    predictions,
    smooth=True,
    extent=pred_ds.scene.extent,
    num_classes=len(class_config))

scores = pred_labels.get_score_arr(pred_labels.extent)
scores_building = scores[0]
scores_background = scores[1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout(w_pad=-2)
ax1.imshow(scores_building, cmap='plasma')
ax1.axis('off')
ax1.set_title('building')
ax2.imshow(scores_background, cmap='plasma')
ax2.axis('off')
ax2.set_title('background')
plt.savefig("plot.png", dpi=800, bbox_inches='tight')
plt.show()

pred_labels.save(
    uri=pred_tif,
    crs_transformer=pred_ds.scene.raster_source.crs_transformer,
    class_config=class_config)


