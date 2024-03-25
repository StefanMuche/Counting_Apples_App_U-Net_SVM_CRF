import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
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

# Directoare
image_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/test/cropped_images1'
output_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/test/predicted_masks1'

# Inițializăm o listă pentru a stoca patch-urile procesate temporar
processed_patches = []

# Procesăm și reasamblăm imaginile
for i, filename in enumerate(sorted(os.listdir(image_dir))):
    if filename.endswith('.jpg'):
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_mask = model(image_tensor)
            pred_mask = torch.sigmoid(pred_mask) > 0.5
            pred_mask_np = pred_mask.cpu().squeeze().numpy().astype(np.uint8) * 255
            processed_patches.append(pred_mask_np)

        # Verificăm dacă am procesat un set de 12 patch-uri
        if (i + 1) % 12 == 0:
            # Reasamblăm cele 12 patch-uri în imaginea originală
            full_image = np.zeros((1024, 768), dtype=np.uint8)
            for j, patch in enumerate(processed_patches):
                row = j // 3 * 256  # Calculăm poziția pe verticală
                col = j % 3 * 256  # Calculăm poziția pe orizontală
                full_image[row:row + 256, col:col + 256] = patch

            # Salvăm imaginea reasamblată
            full_image_pil = Image.fromarray(full_image)
            output_filename = f"reassembled_{i // 11}.png"  # Numărul imaginii reasamblate
            full_image_pil.save(os.path.join(output_dir, output_filename))

            # Resetăm lista de patch-uri procesate pentru următorul set
            processed_patches = []
