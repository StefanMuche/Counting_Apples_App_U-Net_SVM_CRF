import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import cv2
from skimage import color, measure

########## CITIREA UNEI IMAGINI DE TEST SI A IMAGINII SEGMENTATE ######################
ix = 220

mask = cv2.imread(f'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/test/predicted_masks_best_unet_model/predicted_mask_{ix}.jpg')
image = cv2.imread(f'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/test/predicted_images_best_unet_model/test_image{ix}.jpg')

############ PRELUCRAREA IMAGINII SI REALIZAREA NUMARARII MERELOR ####################

# Convertim masca în tonuri de gri și aplicăm un prag pentru a o face binară

img_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

# Etichetăm obiectele conectate din imagine
labels = measure.label(opening, background=0)

# Numărăm etichetele unice, excluzând fundalul (eticheta 0)
unique_labels = np.unique(labels)
# Fundalul este eticheta 0, deci scădem 1 pentru a-l exclude din numărătoare
num_mere = len(unique_labels) - 1

###### AFISARE IMAGINE SEGMENTATA SI NUMARUL DE MERE DIN IMAGINE ################

print(f"Numărul de mere în masca segmentată: {num_mere}")
cv2.imshow('Test image', image)
cv2.imshow('Mask', mask)
cv2.imshow('Segmented Mask with Labels', color.label2rgb(labels, image, bg_label=0, alpha=0.3))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create the RGB overlay image using label2rgb
# labeled_image = color.label2rgb(labels, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), bg_label=0, alpha=0.3)

# # Display the original image
# plt.figure(figsize=(6, 6))
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title('Test Image')
# plt.axis('off')
# plt.show(block=False)  # block=False allows the code to continue without closing the figure

# # Display the opening image (processed image)
# plt.figure(figsize=(6, 6))
# plt.imshow(opening, cmap='gray')
# plt.title('Opening Image')
# plt.axis('off')
# plt.show(block=False)

# # Display the labeled overlay image
# plt.figure(figsize=(6, 6))
# plt.imshow(labeled_image)
# plt.title('Labeled Overlay Image')
# plt.axis('off')
# plt.show(block=False)

# plt.show()
x = np.load('train_loss.npy')
y = np.load('train_acc_.npy')
print(x)

















