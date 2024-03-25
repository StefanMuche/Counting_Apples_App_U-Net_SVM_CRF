import os
import numpy as np
import cv2
from skimage import color, measure
import matplotlib.pyplot as plt

# Paths to the directories
masks_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/test/masks'
images_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/test/images'
output_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/test/labels'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over each image in the masks directory
for filename in os.listdir(masks_dir):
    if filename.endswith('.png'):
        mask_path = os.path.join(masks_dir, filename)
        image_path = os.path.join(images_dir, filename)
        
        mask = cv2.imread(mask_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        
        # Convert the mask to grayscale and apply a binary threshold
        img_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        
        # Apply morphological opening
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
        
        # Label connected components
        labels = measure.label(opening, background=0)
        
        # Count the unique labels, excluding the background
        unique_labels = np.unique(labels)
        num_mere = len(unique_labels) - 1  # Exclude background label
        
        # Overlay the labels on the original image with transparency
        label_overlay = color.label2rgb(labels, image=image, bg_label=0, alpha=0.3)
        
        # Save the output image
        plt.imsave(os.path.join(output_dir, filename), label_overlay)
        
        print(f'Processed {filename}: found {num_mere} apples.')

print("Processing completed.")



###### AFISARE IMAGINE SEGMENTATA SI NUMARUL DE MERE DIN IMAGINE ################

# print(f"Numărul de mere în masca segmentată: {num_mere}")
# cv2.imshow('Test image', image)
# cv2.imshow('Mask', mask)
# cv2.imshow('Segmented Mask with Labels', color.label2rgb(labels, image, bg_label=0, alpha=0.3))
# cv2.waitKey(0)
# cv2.destroyAllWindows()



















