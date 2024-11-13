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
test_img_uri = "/home/nicku/PycharmProjects/pythonProject/ma_dataset/hz.tif"

pred = '/home/nicku/PycharmProjects/pythonProject/results'
pred_bw = '/home/nicku/PycharmProjects/pythonProject/results'
pred_overlap = '/home/nicku/PycharmProjects/pythonProject/results'
pred_tif = '/home/nicku/PycharmProjects/pythonProject/results'
learner_dir = '/home/nicku/PycharmProjects/pythonProject/model'
saved_model = '/home/nicku/PycharmProjects/pythonProject/model/model-bundle.zip'



img_channels = 3
num_workers=2
batch_sz=4
test_batch_sz=4
lr=6e-5
num_epochs = 100
#test_num_epochs = 100
#overfit_num_steps=2
#ignore_class_index=None
#external_loss_def=None
#backbone=Backbone.resnet101
#pretrained = True



class_config = ClassConfig(
    names=['background', 'building'],
    colors=['black', 'white'],
    null_class='background')
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
    A.CoarseDropout(max_height=32, max_width=32, max_holes=1)
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
    size=200,
    stride=100,
    transform=A.Resize(256, 256))

#x,y = train_ds[0]
#x2,y2 = val_ds[0]

#print(x.shape, y.shape)
#print(x2.shape, y2.shape)

#viz.plot_batch(x.unsqueeze(0), y.unsqueeze(0), show=True)
#x, y = viz.get_batch(train_ds,4)
#viz.plot_batch(x,y, show=True)

data_cfg = SemanticSegmentationGeoDataConfig(
    img_channels = img_channels,
    class_names=class_config.names,
    class_colors=class_config.colors,
    num_workers=num_workers,  # increase to use multi-processing
)

solver_cfg = SolverConfig(
    batch_sz=batch_sz,
    #test_batch_sz=test_batch_sz,
    lr=lr,
    class_loss_weights=[0.5, 0.5],#[1., 10.], #[0.5, 0.5], #
    num_epochs = num_epochs,
    #test_num_epochs = test_num_epochs,
    #overfit_num_steps=overfit_num_steps,
    #ignore_class_index=ignore_class_index,
    #external_loss_def=external_loss_def,
)

model = SemanticSegmentationModelConfig(backbone=Backbone.resnet101, pretrained = True, init_weights=None)
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



learner_cfg = SemanticSegmentationLearnerConfig(data=data_cfg, solver=solver_cfg , model=model)


learner = SemanticSegmentationLearner(
    cfg=learner_cfg,
    output_dir=learner_dir,
    train_ds=train_ds,
    valid_ds=val_ds,
    #model=model

)
learner.train()
#learner.plot_predictions(split='valid', show=True) # if you want to use valid as pred
learner.save_model_bundle()

"""learner = SemanticSegmentationLearner.from_model_bundle(
    model_bundle_uri=saved_model,
    output_dir=pred,
    model=model,

    #if you want to continue training, on same dataset or on a different dataset.
    #provide paths of train_ds and val_ds of different dataset incase of different dataset
    #set training true in that case

    #train_ds=train_ds,
    #valid_ds=val_ds,
    #training=True,
    training = False
)

#train more if you agree to the learner = SemanticSegmentationLearner.from_model_bundle
#learner.train()"""

class customSemanticSegmentationDataReader(Dataset):
    """Reads semantic segmentatioin images and labels from files."""

    def __init__(self, img_dir: str):#, label_dir: str):
        """Constructor.

        Args:
            img_dir (str): Directory containing images.
            label_dir (str): Directory containing segmentation masks.
        """
        self.img_dir = Path(img_dir)
        #self.label_dir = Path(label_dir)

        # collect image and label paths, match them based on filename
        img_paths = discover_images(img_dir)
        #label_paths = discover_images(label_dir)
        self.img_paths = sorted(img_paths, key=lambda p: p.stem)
        #self.label_paths = sorted(label_paths, key=lambda p: p.stem)
        #self.validate_paths()

    """def validate_paths(self) -> None:
        if len(self.img_paths) != len(self.label_paths):
            raise ImageDatasetError(
                'There should be a label file for every image file. '
                f'Found {len(self.img_paths)} image files and '
                f'{len(self.label_paths)} label files.')
        for img_path, label_path in zip(self.img_paths, self.label_paths):
            if img_path.stem != label_path.stem:
                raise ImageDatasetError(
                    f'Name mismatch between image file {img_path.stem} '
                    f'and label file {label_path.stem}.')"""

    def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.img_paths[ind]
        #label_path = self.label_paths[ind]

        x = load_image(img_path)
        #y = None# load_image(label_path).squeeze()

        return x #, y

    def __len__(self):
        return len(self.img_paths)


class customSemanticSegmentationImageDataset(ImageDataset):
    """Reads semantic segmentatioin images and labels from files.

    Uses :class:`.SemanticSegmentationDataReader` to read the data.
    """

    def __init__(self, img_dir: str, *args, **kwargs):
        """Constructor.

        Args:
            img_dir (str): Directory containing images.
            label_dir (str): Directory containing segmentation masks.
            *args: See :meth:`.ImageDataset.__init__`.
            **kwargs: See :meth:`.ImageDataset.__init__`.
        """

        ds = customSemanticSegmentationDataReader(img_dir)#, label_dir)
        super().__init__(
            ds,
            *args,
            **kwargs,
            transform_type=TransformType.semantic_segmentation)


"""ds = customSemanticSegmentationImageDataset( #if your test dataset have tiles #also use the custom semanticsegmentation functions
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

"""scene = make_ss_scene( # we dont need this as we dont have test labels and we dont need to see the prediction scores
    class_config=class_config,
    image_uri=test_img_uri,
    label_vector_uri=test_label_uri,
    label_vector_default_class_id=class_config.get_class_id('building'),
    label_raster_source_kw=dict(
        background_class_id=class_config.get_class_id('background')),
    image_raster_source_kw=dict(allow_streaming=True))

gt_labels = scene.label_source.get_labels()

evaluator = SemanticSegmentationEvaluator(class_config)

evaluation = evaluator.evaluate_predictions(
    ground_truth=gt_labels, predictions=pred_labels)
print("Evaluation for the building class:",evaluation.class_to_eval_item[0])
print("Evaluation for the background class:",evaluation.class_to_eval_item[1])
"""

"""# Open the predicted GeoTIFF file
with rasterio.open("/home/nicku/PycharmProjects/pythonProject/results/labels.tif") as src:
    # Read the raster data
    raster_data = src.read()

# The first band should be the 'building' class
building_footprints = raster_data[0]

# Convert the building footprints to a binary black and white image
building_footprints_bw = np.where(building_footprints > 0, 255, 0).astype(np.uint8)

# Save the black and white image
Image.fromarray(building_footprints_bw).save(pred_bw)


# Load the original test image
with rasterio.open(test_img_uri) as src:
    test_img = src.read()

# Create a RGB image with the predicted building footprints
building_footprints_rgb = np.stack([building_footprints_bw]*3, axis=-1)

# Overlay the predicted footprints on the original image with transparency
overlay_img = test_img * 0.7 + building_footprints_rgb * 0.3

# Save the RGB overlay image
Image.fromarray(overlay_img.astype(np.uint8)).save(pred_overlap)

# Define a color map for the binary image
cmap = ListedColormap(['black', 'white'])

# Display the binary black and white image
plt.imshow(building_footprints_bw, cmap=cmap)
plt.axis('off')
plt.savefig(pred_bw, bbox_inches='tight', pad_inches = 0)

# You may need to adjust the factor of 0.7 and 0.3 depending 
#on how transparent you want the overlaid image to be."""
