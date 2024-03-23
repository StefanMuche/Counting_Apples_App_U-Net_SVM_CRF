import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from unet_pytorch import UNet

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

# Directorul cu imagini și directorul destinație
image_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/test/cropped_images1'
output_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/ProcessedImages'

# Procesează fiecare imagine și reasamblează
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):  # Presupunem că imaginile sunt în format .jpg
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Obține predicția
        with torch.no_grad():
            pred_mask = model(image_tensor)
            pred_mask = torch.sigmoid(pred_mask) > 0.5  # Transformă logitii în probabilități și aplică un prag

        # Transformă tensorul înapoi în imagine
        pred_mask_np = pred_mask.cpu().squeeze().numpy().astype(np.float32)
        pred_image = Image.fromarray((pred_mask_np * 255).astype(np.uint8))

        # Salvează imaginea procesată (aici ar trebui ajustat pentru a salva toate bucățile)
        pred_image.save(os.path.join(output_dir, filename))

# Acum trebuie să adaugi logica pentru reasamblarea celor 12 bucăți într-o imagine mai mare de 768x1024
# Acest pas depinde de modul în care sunt ordonate și stocate imaginile segmentate
# De exemplu, dacă le-ai stocat într-o ordine specifică, va trebui să le încarci și să le așezi corespunzător într-o matrice
