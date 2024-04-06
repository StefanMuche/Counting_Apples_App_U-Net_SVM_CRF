import os
import random
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

TRAIN_IMAGES_PATH = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/resized_images1'

images = os.listdir(TRAIN_IMAGES_PATH)

random_img_name = random.choice(images)
img_path = os.path.join(TRAIN_IMAGES_PATH, 'resized_image_556.jpg')

img = io.imread(img_path)
hsv_img = color.rgb2hsv(img)

brightness_increase = 0.5

hsv_img_brighter = hsv_img.copy()
hsv_img_brighter[:,:,2] = np.clip(hsv_img[:,:,2] + brightness_increase, 0, 1)
img_brighter = color.hsv2rgb(hsv_img_brighter)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(color.hsv2rgb(hsv_img))
plt.title('Original HSV Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_brighter)
plt.title('Brighter Image')
plt.axis('off')

plt.show()

#####################################################################
#################### CONSTRUIREA GRAFULUI ###########################
#####################################################################

def create_graph(img_brighter):
    h, w = img_brighter.shape[:2]
    p = np.zeros((h, w), dtype=int)
    c = {}

    for i in range(h):
        for j in range(w):
            pixel_rgb = img_brighter[i, j, :]
            p[i, j] = 1 if pixel_rgb[0] * 255 > 110 and pixel_rgb[1] * 255 < 80 else 0
    
    for i in range(h):
        for j in range(w):
            c[(i, j)] = {}
            if j+1 < w: 
                # Verifică dacă pixelii sunt identici și fac parte din componenta măr
                c[(i, j)][(i, j+1)] = 0 if p[i, j] == 1 and p[i, j] == p[i, j+1] else 1
            if i+1 < h:  
                # Verifică dacă pixelii sunt identici și fac parte din componenta măr
                c[(i, j)][(i+1, j)] = 0 if p[i, j] == 1 and p[i, j] == p[i+1, j] else 1
    return p, c


p, c = create_graph(img_brighter)

##################################################################################
############### PARCUGEREA GRAFULUI SI SEGMENTAREA IMAGINII ######################
##################################################################################
import numpy as np

def segment_apples1(p, c):
    h, w = p.shape
    a = np.zeros((h, w), dtype=int)
    
    for i in range(h):
        for j in range(w):
            if p[i, j] == 1:
                a[i, j] = 1  # Setează pixelul curent dacă este parte din măr
                if j + 1 < w and c[(i, j)].get((i, j+1), 1) == 0:
                    a[i, j+1] = 1
                if i + 1 < h and c[(i, j)].get((i+1, j), 1) == 0:
                    a[i+1, j] = 1
            # Nu mai este necesar else-ul de la sfârșit, deoarece a este inițializat cu 0 pentru toți pixelii

    return a

segmented_image = segment_apples1(p, c)

# Crearea unei imagini segmentate unde pixelii din măr sunt păstrați de la img_brighter
segmented_display = img.copy()
# Pentru pixelii care nu sunt parte din măr, îi setăm la o culoare neutru, de exemplu gri
segmented_display[segmented_image == 0] = [0.5, 0.5, 0.5]  # Setează gri pentru fundal

# Afișarea imaginii originale și a celei segmentate
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Brighter Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_display)
plt.title('Segmented Image (Original Apples Preserved)')
plt.axis('off')

plt.show()

##################################################################################
######################### AFISAREA IMAGINII SEGMENTATE ###########################
##################################################################################


