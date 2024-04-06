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
from skimage import color, measure

########### DEFINIREA CONSTANTELOR ######################

PATCH_SIZE= 256
NR_PATCHES = 12
DIM_TRAIN_IMAGES = 331
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

transform = transforms.Compose([
    transforms.ToTensor(),
])

########## INCARCAREA MODELULUI ########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(pretrained=True, out_channels=1)
model_path = "D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/modele/unet_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

############# DEFINIREA VECTORILOR CE CONTIN IMAGINILE ##############################

x_train_uncropped = np.zeros((DIM_TRAIN_IMAGES, 1024, 768, IMG_CHANNELS), dtype = np.uint8)

def process_and_reconstruct_image(image_path, model, device, transform):
   
    # Încarcă imaginea originală
    
    image = Image.open(image_path)
    image = image.convert('RGB')  #Imaginea este în modul RGB

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
    image = cv2.imread(image_path)
    image = cv2.resize(image, (768, 1024))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR in RGB
    
    # Convertesc masca din Pillow Image într-un array NumPy
    reconstructed_mask_np = np.array(reconstructed_mask_image)

    # Acum, folosind array-ul NumPy, putem converti imaginea în nuanțe de gri cu OpenCV
    # img_gray = cv2.cvtColor(reconstructed_mask, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(reconstructed_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    
    
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    
    
    labels = measure.label(opening, background=0)
    
    #Numara toate label-urile unice
    unique_labels = np.unique(labels)
    num_mere = len(unique_labels) - 1  # Elimina fundalul negru din numarare
    
    #Afiseaza label-urile peste imaginea originala convertita in gri
    label_overlay = color.label2rgb(labels, image=image, bg_label=0, alpha=0.3)
    print(f'Found {num_mere} apples.')

    return label_overlay, num_mere


