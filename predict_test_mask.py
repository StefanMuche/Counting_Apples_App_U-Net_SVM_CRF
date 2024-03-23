import numpy as np
import os
from tqdm import tqdm
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from unet_pytorch import UNet
import matplotlib.pyplot as plt


########### DEFINIREA CONSTANTELOR ######################

PATCH_SIZE= 256
NR_PATCHES = 12
DIM_TRAIN_IMAGES = 331
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

########### DEFINIREA CAILOR CATRE IMAGINI SI MASTI ##################

TEST_PATH =  'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/test/images'
SAVE_PATH = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/test/masks'

IMAGES = os.listdir(TEST_PATH)

# Definește transformarea
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Încărcarea modelului
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(pretrained=True, out_channels=1)
model_path = "D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/modele/unet_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

############# DEFINIREA VECTORILOR CE CONTIN IMAGINILE ##############################

x_train_uncropped = np.zeros((DIM_TRAIN_IMAGES, 1024, 768, IMG_CHANNELS), dtype = np.uint8)


def process_and_reconstruct_image(image_path, model, device, transform, save_path_mask):
    # Încarcă imaginea originală
    
    image = Image.open(image_path)
    image = image.convert('RGB')  # Asigură-te că imaginea este în modul RGB

    image = image.resize((768, 1024), Image.BILINEAR)
    
    # Inițializează tensorul pentru imaginea reconstituită
    reconstructed_mask = np.zeros((1024, 768), dtype=np.uint8)

    # Imparte imaginea în patch-uri și procesează fiecare patch
    for i in range(4):  # 4 patch-uri pe înălțime
        for j in range(3):  # 3 patch-uri pe lățime
            # Extrage patch-ul
            patch = image.crop((j * PATCH_SIZE, i * PATCH_SIZE, (j+1) * PATCH_SIZE, (i+1) * PATCH_SIZE))
            
            # Transformă și adaugă dimensiunea batch-ului
            patch_tensor = transform(patch).unsqueeze(0).to(device)
            
            # Aplică modelul pe patch
            with torch.no_grad():
                pred_mask = model(patch_tensor)
                pred_mask = torch.sigmoid(pred_mask) > 0.5
                pred_mask_np = pred_mask.cpu().squeeze().numpy().astype(np.uint8) * 255
            
            # Adaugă patch-ul procesat în masca reconstituită
            reconstructed_mask[i*PATCH_SIZE:(i+1)*PATCH_SIZE, j*PATCH_SIZE:(j+1)*PATCH_SIZE] = pred_mask_np
    
    # Converteste masca reasamblată înapoi într-un obiect Image pentru afișare
    reconstructed_mask_image = Image.fromarray(reconstructed_mask)

    reconstructed_mask_image.save(save_mask_path)
    
    # # Afișează imaginea originală și masca reasamblată
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image)
    # plt.title('Imagine Originală')
    # plt.axis('off')
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(reconstructed_mask_image, cmap='gray')
    # plt.title('Mască Reasamblată')
    # plt.axis('off')
    
    # plt.show()

# # Aplică funcția pe o imagine de test
# image_path = os.path.join(TEST_PATH, IMAGES[153])  # Presupunem că IMAGES[0] este validă și există
# process_and_reconstruct_image(image_path, model, device, transform)


for img_name in IMAGES:
    image_path = os.path.join(TEST_PATH, img_name)
    save_mask_path = os.path.join(SAVE_PATH, img_name)  # Assuming you want to save with the same name
    # Modify 'process_and_reconstruct_image' to accept 'save_path' and use it to save the mask.
    process_and_reconstruct_image(image_path, model, device, transform, save_mask_path)


