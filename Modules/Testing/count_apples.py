import numpy as np
import os
from tqdm import tqdm
import cv2
import torch
from torchvision import transforms
from PIL import Image
from unet_pytorch import UNet
import matplotlib.pyplot as plt
from skimage import color, measure
from sklearn.metrics import accuracy_score, mean_squared_error
import torch.nn.functional as F
from skimage import io
########### DEFINIREA CONSTANTELOR ######################

PATCH_SIZE = 256
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

images_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images_for_errors_png'
masks_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/Masks_for_errors'

def process_and_reconstruct_image(image_path, model, device, transform):
    # Încarcă imaginea originală
    image = Image.open(image_path)
    image = image.convert('RGB')  # Imaginea este în modul RGB

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
    
    return reconstructed_mask

def calculate_metrics(segmented_image, original_mask):
    # Flatten the images
    segmented_flat = segmented_image.flatten()
    original_flat = original_mask.flatten()

    # Calculate accuracy
    accuracy = accuracy_score(original_flat, segmented_flat)
    
    # Calculate mean squared error (MSE)
    mse = mean_squared_error(original_flat, segmented_flat)

    # Convert arrays to tensors for loss calculations
    segmented_tensor = torch.tensor(segmented_flat, dtype=torch.float32)
    original_tensor = torch.tensor(original_flat, dtype=torch.float32)

    # Calculate Binary Cross-Entropy Loss (BCE Loss)
    bce_loss = F.binary_cross_entropy_with_logits(segmented_tensor, original_tensor).item()

    # Calculate Dice Loss
    smooth = 1e-5
    intersection = (segmented_tensor * original_tensor).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (segmented_tensor.sum() + original_tensor.sum() + smooth)

    return accuracy, mse, bce_loss, dice_loss.item()

def process_and_evaluate_all_images(images_dir, masks_dir, model, device, transform):
    accuracies = []
    mses = []
    bce_losses = []
    dice_losses = []

    for filename in tqdm(os.listdir(images_dir)):
        if filename.endswith('.png'):  # Adjust if images are not in PNG format
            image_path = os.path.join(images_dir, filename)
            mask_path = os.path.join(masks_dir, filename)

            try:
                # Segment the image
                segmented_image = process_and_reconstruct_image(image_path, model, device, transform)

                # Load and resize the original mask
                original_mask = io.imread(mask_path)
                if original_mask.shape[2] == 4:
                    original_mask = original_mask[:, :, :3]
                if len(original_mask.shape) == 3:
                    original_mask = color.rgb2gray(original_mask) * 255
                original_mask = cv2.resize(original_mask.astype(np.uint8), (768, 1024))

                # Calculate metrics
                accuracy, mse, bce_loss, dice_loss = calculate_metrics(segmented_image, original_mask)
                accuracies.append(accuracy)
                mses.append(mse)
                bce_losses.append(bce_loss)
                dice_losses.append(dice_loss)
                
                print(f'Processed {filename}: Accuracy = {accuracy}, MSE = {mse}, BCE Loss = {bce_loss}, Dice Loss = {dice_loss}')
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Calculate and print overall metrics
    overall_accuracy = np.mean(accuracies)
    overall_mse = np.mean(mses)
    overall_bce_loss = np.mean(bce_losses)
    overall_dice_loss = np.mean(dice_losses)
    
    print(f'Overall Accuracy: {overall_accuracy}')
    print(f'Overall Mean Squared Error: {overall_mse}')
    print(f'Overall Binary Cross-Entropy Loss: {overall_bce_loss}')
    print(f'Overall Dice Loss: {overall_dice_loss}')

    return overall_accuracy, overall_mse, overall_bce_loss, overall_dice_loss

# Run the evaluation
overall_accuracy, overall_mse, overall_bce_loss, overall_dice_loss = process_and_evaluate_all_images(images_dir, masks_dir, model, device, transform)
