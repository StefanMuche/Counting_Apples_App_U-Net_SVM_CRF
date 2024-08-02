import torch
import os
from tqdm import tqdm
import cv2
import numpy as np

###### DEFINIREA CONSTANTELOR PENTRU DIMENSIUNI SI CAI DE ACCES ############################

PATCH_SIZE = 256
DIM_TRAIN_IMAGES = 670
DIM_TRAIN_MASKS = 670
IMG_CHANNELS = 3


SAVE_PATH_IMAGES = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/resized_images1'
SAVE_PATH_MASKS = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/resized_masks1'
IMAGES_PATCHES = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/cropped_images1'
MASKS_PATCHES = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/cropped_masks1'
TRAIN_IMAGES_PATH = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images'
TRAIN_MASKS_PATH = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/masks'
TEST_PATH =  'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/test/images'

IMAGES = os.listdir(TRAIN_IMAGES_PATH)
MASKS =  os.listdir(TRAIN_MASKS_PATH)


x_train_uncropped = torch.zeros(DIM_TRAIN_IMAGES, 1024, 768, IMG_CHANNELS, dtype=torch.uint8)
y_train_uncropped = torch.zeros(DIM_TRAIN_MASKS, 1024, 768, 1, dtype=torch.bool)


os.makedirs(IMAGES_PATCHES, exist_ok=True)
os.makedirs(MASKS_PATCHES, exist_ok=True)
os.makedirs(SAVE_PATH_IMAGES, exist_ok=True)
os.makedirs(SAVE_PATH_MASKS, exist_ok=True)

########### REDIMENSIONAREA IMAGINILOR SI MASTILOR LA DIMENSIUNEA 768X1024 ###############

def reshape_train_data(images_path, masks_path, x_data, y_data):
    print("Redimensionarea setului de date...\n")

    for n, image_file in enumerate(tqdm(os.listdir(images_path))):
        image_path = os.path.join(images_path, image_file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (768, 1024))
        x_data[n] = torch.from_numpy(image)
    

    for n, mask_file in enumerate(tqdm(os.listdir(masks_path))):
        mask_path = os.path.join(masks_path, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (768, 1024))
        mask = np.expand_dims(mask, axis=-1)
        y_data[n] = torch.from_numpy(mask).bool()

    print("Redimensionarea setului de date a fost efectuata!\n")
    return x_data, y_data

########## IMPARTIREA SETULUI DE ANTRENAMENT IN PATCH-URI DE 256x256 ##################

def split_images_masks_torch(x_data, y_data, patch_size):
    print("Impartirea imaginilor si mastilor in pachete...\n")
    num_images, img_height, img_width, _ = x_data.shape
    patch_per_image = (img_height // patch_size) * (img_width // patch_size)
    x_patches = torch.zeros(num_images * patch_per_image, patch_size, patch_size, IMG_CHANNELS, dtype=torch.uint8)
    y_patches = torch.zeros(num_images * patch_per_image, patch_size, patch_size, 1, dtype=torch.bool)
    
    index = 0
    for i in range(num_images):
        for row in range(0, img_height - patch_size + 1, patch_size):
            for col in range(0, img_width - patch_size + 1, patch_size):
                x_patches[index] = x_data[i, row:row + patch_size, col:col + patch_size]
                y_patches[index] = y_data[i, row:row + patch_size, col:col + patch_size]
                index += 1
    
    print("Impartirea imaginilor si mastilor in pachete a fost realizata!\n")
    return x_patches, y_patches

########## SALVAREA PATCH-URILOR ###################

def save_patches_torch(images, masks, save_path_images, save_path_masks):
    print("Salvarea bucatilor de imagini...\n")
    for i in range(images.shape[0]):
        image_filename = f"patched_image_{i + 1}.jpg"
        mask_filename = f"patched_mask_{i + 1}.jpg"
        image_path = os.path.join(save_path_images, image_filename)
        mask_path = os.path.join(save_path_masks, mask_filename)
        cv2.imwrite(image_path, images[i].numpy())
        cv2.imwrite(mask_path, masks[i].numpy().astype(np.uint8) * 255)
    print("Salvarea bucatilor de imagini a fost efectuata!\n")


x_train_uncropped, y_train_uncropped = reshape_train_data(TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH, x_train_uncropped, y_train_uncropped)
x_train_patches, y_train_patches = split_images_masks_torch(x_train_uncropped, y_train_uncropped, PATCH_SIZE)
save_patches_torch(x_train_patches, y_train_patches, IMAGES_PATCHES, MASKS_PATCHES)