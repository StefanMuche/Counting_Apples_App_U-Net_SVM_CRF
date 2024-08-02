import os
import numpy as np
import cv2
from skimage import color, measure
import matplotlib.pyplot as plt

############ CAILE CATRE IMAGINI SI MASTI ###############################

masks_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/Masks_for_errors'
images_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images_for_errors_png'
output_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/Labels_train_data'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterez prin fiecare masca 
for filename in os.listdir(masks_dir):
    if filename.endswith('.png'):
        mask_path = os.path.join(masks_dir, filename)
        image_path = os.path.join(images_dir, filename)
        
        mask = cv2.imread(mask_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (768, 1024))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR in RGB
        
        # Convertesc masca din color in gri si aplic un threshold binar
        img_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        
        
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
        
        
        labels = measure.label(opening, background=0)
        
        #Numara toate label-urile unice
        unique_labels = np.unique(labels)
        num_mere = len(unique_labels) - 1  # Elimina fundalul negru din numarare
        
        #Afiseaza label-urile peste imaginea originala convertita in gri
        label_overlay = color.label2rgb(labels, image=image, bg_label=0, alpha=0.3)
        
        # Salvez imaginea
        plt.imsave(os.path.join(output_dir, filename), label_overlay)
        
        print(num_mere)

print("Processing completed.")
# Example usage
