import datetime

import torch
import pandas as pd
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch_snippets import *
import torch.nn as nn
from torchvision.models import vgg16_bn
import warnings
warnings.filterwarnings("ignore")


IMG_ROOT = 'C:/Users/DELL G3/PycharmProjects/autonomous/dataset/'
df = pd.read_csv('C:/Users/DELL G3/PycharmProjects/autonomous/dataset/metadata.csv')

train_df = df.loc[df['split'] == 'train']
tst_df = df.loc[df['split'] == 'test']
val_df = df.loc[df['split'] =='val']
print("Training dataset size:", len(train_df))
print("Test dataset size:", len(tst_df))
print("Validation dataset size:", len(val_df))

tfms = transforms.Compose([
    #     transforms.RandomHorizontalFlip(0.3),
    #     transforms.RandomRotation(40),
    #     transforms.RandomVerticalFlip(0.3),
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406],  # imagenet
    #                     [0.229, 0.224, 0.225])
])


class BuildingDataset(Dataset):
    def __init__(self, df, root=IMG_ROOT):
        self.df = df
        self.root = root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        w, h = 256, 256
        img_path = self.root + self.df.iloc[idx]['png_image_path']
        mask_path = self.root + self.df.iloc[idx]['png_label_path']
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        img = cv2.resize(img, (w, h))
        mask = cv2.resize(mask, (w, h))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return img, mask

    def collate_fn(self, batch):
        images, masks = list(zip(*batch))
        images = torch.cat([tfms(img.copy() / 255.)[None] for img in images]).float().to(device)
        masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long().to(device)
        return images, masks


tr_ds = BuildingDataset(train_df)
tst_ds = BuildingDataset(tst_df)
val_ds = BuildingDataset(val_df)


tr_dl = DataLoader(tr_ds, batch_size=4, drop_last=True, shuffle=True,
                   collate_fn=tr_ds.collate_fn)
val_dl = DataLoader(tr_ds, batch_size=1, drop_last=True, shuffle=False,
                    collate_fn=val_ds.collate_fn)
tst_dl = DataLoader(tr_ds, batch_size=1, drop_last=False, shuffle=False,
                    collate_fn=tst_ds.collate_fn)

n_epochs = 5
#img, mask = tr_ds[10]
#fig, ax = plt.subplots(1, 2)
#show(img, ax=ax[0])
#show(mask, ax=ax[1])

def conv(in_channels, out_channels):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3,
                stride=1, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
  )

def up_conv(in_channels, out_channels):
    return nn.Sequential(
     nn.ConvTranspose2d(in_channels, out_channels,
                         kernel_size=2, stride=2),
     nn.ReLU(inplace=True)
  )

class UNet(nn.Module):
    def __init__(self, weights=True, out_channels=12):
        super().__init__()
        self.backbone = vgg16_bn(weights=True).to(device).features
        self.down1 = nn.Sequential(*self.backbone[:6]) # 64
        self.down2 = nn.Sequential(*self.backbone[6:13]) # 128
        self.down3 = nn.Sequential(*self.backbone[13:20]) # 256
        self.down4 = nn.Sequential(*self.backbone[20:27]) # 512
        self.down5 = nn.Sequential(*self.backbone[27:34]) # 512

        self.bottleneck = nn.Sequential(*self.backbone[34:]) # 512
        self.conv_bottleneck = conv(512, 1024)

        self.up_conv5 = up_conv(1024, 512)
        self.merge_conv5 = conv(512+512, 512)
        self.up_conv4 = up_conv(512, 256)
        self.merge_conv4 = conv(512 + 256, 256)
        self.up_conv3 = up_conv(256, 128)
        self.merge_conv3 = conv(256+128, 128)
        self.up_conv2 = up_conv(128, 64)
        self.merge_conv2 = conv(128+64, 64)
        self.up_conv1 = up_conv(64, 32)
        self.merge_conv1 = conv(32+64, 32)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # encoder
        x_1 = self.down1(x)
        x_2 = self.down2(x_1)
        x_3 = self.down3(x_2)
        x_4 = self.down4(x_3)
        x_5 = self.down5(x_4)
        # bottleneck
        x = self.bottleneck(x_5)
        x = self.conv_bottleneck(x)
        # decoder
        x = self.up_conv5(x)
        x = self.merge_conv5(torch.cat([x, x_5], dim=1))
        x = self.up_conv4(x)
        x = self.merge_conv4(torch.cat([x, x_4], dim=1))
        x = self.up_conv3(x)
        x = self.merge_conv3(torch.cat([x, x_3], dim=1))
        x = self.up_conv2(x)
        x = self.merge_conv2(torch.cat([x, x_2], dim=1))
        x = self.up_conv1(x)
        x = self.merge_conv1(torch.cat([x, x_1], dim=1))

        x = self.final_conv(x)
        return x

model = UNet().to(device)
model(torch.zeros((1, 3, 256, 256)).to(device))


def loss_fn(preds, targets):
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(preds, targets)
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    return loss, acc


def train_batch(model, batch, optim, loss_fn):
    model.train()
    imgs, masks = batch
    pred_masks = model(imgs)
    optim.zero_grad()
    loss, acc = loss_fn(pred_masks, masks)
    loss.backward()
    optim.step()
    return loss.item(), acc.item()


@torch.no_grad()
def validate_batch(model, batch, loss_fn):
    model.eval()
    imgs, masks = batch
    pred_masks = model(imgs)
    loss, acc = loss_fn(pred_masks, masks)
    return loss.item(), acc.item()


model = UNet(out_channels=300).to(device)
optim = torch.optim.Adam(model.parameters(), lr=6e-5)
loss_fn = loss_fn

log = Report(n_epochs)
for e in range(n_epochs):
    N = len(tr_dl)
    for i, batch in enumerate(tr_dl):
        loss, acc = train_batch(model, batch, optim, loss_fn)
        log.record(e + (i + 1) / N, trn_loss=loss, trn_acc=acc, end='\r')
    N = len(tst_dl)
    for i, batch in enumerate(tst_dl):
        loss, acc = validate_batch(model, batch, loss_fn)
        log.record(e + (i + 1) / N, val_loss=loss, val_acc=acc, end='\r')

    log.report_avgs(e + 1)

log.plot_epochs(['trn_loss', 'val_loss'])
log.plot_epochs(['trn_acc', 'val_acc'])


im, mask = next(iter(val_dl))
test_mask = model(im)
_, test_mask = torch.max(test_mask, dim=1)
plt.figure(dpi = 800)
fig, ax = plt.subplots(1, 3, dpi = 800)
show(im[0].permute(1,2,0).detach().cpu()[:,:,0], ax= ax[0])
show(mask.permute(1,2,0).detach().cpu()[:,:,0], ax = ax[1])
show(test_mask.permute(1,2,0).detach().cpu()[:,:,0], ax = ax[2])

filename = datetime.datetime.now().strftime("plot_%Y-%m-%d_%H-%M-%S.png")
save_location = "C:/Users/DELL G3/PycharmProjects/autonomous/dataset/png/predict/"
plt.savefig(save_location + filename)