import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg16_bn
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


class FruitSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace("image", "mask"))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image,mask

def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, pretrained=True, out_channels=12):
        super().__init__()

        # Definim blocurile encoderului cu max pooling
        self.block1 = conv(3, 32)
        self.block11 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = conv(32, 64)
        self.block22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = conv(64, 128)
        self.block33 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block4 = conv(128, 256)
        self.block44 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck fără max pooling
        self.conv_bottleneck = conv(256, 512)

        # Definim blocurile decoderului
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = conv(256 + 256, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = conv(128 + 128, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = conv(64 + 64, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = conv(32 + 32, 32)
        self.conv11 = nn.Sequential(nn.Conv2d(32, out_channels, kernel_size=1))

    def forward(self, x):
        # print(f': {x.shape}')
        block1 = self.block1(x)
        # print(f'Inainte de maxpool: {block1.shape}')
        block11 = self.block11(block1)
        # print(f': {block11.shape}')
        block2 = self.block2(block11)
        # print(f'Inainte de maxpool: {block2.shape}')
        block22 = self.block22(block2)
        # print(f': {block22.shape}')
        block3 = self.block3(block22)
        # print(f'Inainte de maxpool: {block3.shape}')
        block33 = self.block33(block3)
        # print(f': {block33.shape}')
        block4 = self.block4(block33)
        # print(f'Inainte de maxpool: {block4.shape}')
        block44 = self.block44(block4)
        # print(f': {block44.shape}')
        x = self.conv_bottleneck(block44)
        # print(f': {x.shape}')

        x = self.up_conv7(x)
        # print(f': {x.shape}')
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)
        # print(f': {x.shape}')

        x = self.up_conv8(x)
        # print(f': {x.shape}')
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)
        # print(f': {x.shape}')

        x = self.up_conv9(x)
        # print(f': {x.shape}')
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)
        # print(f': {x.shape}')

        x = self.up_conv10(x)
        # print(f': {x.shape}')
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)
        # print(f': {x.shape}')

        x = self.conv11(x)
        # print(f': {x.shape}')

        return x


# Adjusting the loss function for binary segmentation
def UnetLoss(preds, targets):
    # Using BCEWithLogitsLoss which combines a sigmoid and binary cross entropy loss
    bce_loss = nn.BCEWithLogitsLoss()(preds, targets)
    preds_bin = torch.sigmoid(preds) > 0.5  # Thresholding to get binary predictions
    acc = (preds_bin == targets).float().mean()
    return bce_loss, acc

# Training function for a single batch
def train_batch(model, data, optimizer, criterion):
    model.train()
    ims, masks = data
    ims, masks = ims.to(device), masks.to(device)
    _masks = model(ims)
    optimizer.zero_grad()
    loss, acc = criterion(_masks, masks)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()

# Validation function for a single batch
@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()
    ims, masks = data
    ims, masks = ims.to(device), masks.to(device)
    _masks = model(ims)
    loss, acc = criterion(_masks, masks)
    return loss.item(), acc.item()

if __name__ == '__main__':
    # Now, place your training script here.
    # This includes dataset instantiation, DataLoader creation, model training, etc.

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = FruitSegmentationDataset(
        image_dir='D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/cropped_images1',
        mask_dir='D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/cropped_masks1',
        transform=transform
    )

    val_dataset = FruitSegmentationDataset(
        image_dir='D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/val_images1',
        mask_dir='D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/val_masks1',
        transform=transform
    )

    trn_dl = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Example model instantiation and training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(pretrained=True, out_channels=1).to(device)
    criterion = UnetLoss
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    n_epochs = 15
    best_loss = np.inf
    val_loss_history = []
    val_acc_history = []
    train_loss_history = []
    train_acc_history = []
    for epoch in range(n_epochs):
        model.train()
        train_loss, train_acc = 0, 0
        print(f"Epoch {epoch + 1}/{n_epochs}")
        for data in tqdm(trn_dl):
            loss, acc = train_batch(model, data, optimizer, criterion)
            train_loss += loss
            train_acc += acc
        
        train_loss /= len(trn_dl)
        train_acc /= len(trn_dl)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        
        model.eval()
        val_loss, val_acc = 0, 0
        for data in tqdm(val_dl):
            loss, acc = validate_batch(model, data, criterion)
            val_loss += loss
            val_acc += acc
        
        val_loss /= len(val_dl)
        val_acc /= len(val_dl)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = "D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/modele/unet_model_512.pth"
            torch.save(model.state_dict(), save_path)     
            np.save('train_loss_history_512.npy', np.array(train_loss_history))
            np.save('val_loss_history_512.npy', np.array(val_loss_history))   
            np.save('train_acc_history_512.npy', np.array(train_acc_history))
            np.save('val_acc_history_512.npy', np.array(val_acc_history))   



        
