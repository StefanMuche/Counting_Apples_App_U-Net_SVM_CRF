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

brightness_increase = 0.2

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
                c[(i, j)][(i, j+1)] = 0 if p[i, j] == p[i, j+1] else 1
            if i+1 < h:  
                c[(i, j)][(i+1, j)] = 0 if p[i, j] == p[i+1, j] else 1
            # if i-1 > 0:
            #     c[(i, j)][(i-1, j)] = 0 if p[i, j] == p[i-1, j] else 1
            # if j-1 > 0:
            #     c[(i, j)][(i, j-1)] = 0 if p[i, j] == p[i, j-1] else 1
    return p, c

p, c = create_graph(img_brighter)

##################################################################################
############### PARCUGEREA GRAFULUI SI SEGMENTAREA IMAGINII ######################
##################################################################################

def segment_apples(p, c):
    h, w = p.shape
    S_tree = {}
    T_tree = {}
    unvisited = set((i, j) for i in range(h) for j in range(w))
    
    while unvisited:
        LM = next(iter(unvisited))
        queue = [LM]
        unvisited.remove(LM)
        S_tree[LM] = []
        
        while queue:
            current = queue.pop(0)
            if current not in S_tree:
                S_tree[current] = []
            
            if current in c:
                neighbors = [neighbor for neighbor in c[current] if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w]
                for neighbor in neighbors:
                    if neighbor in unvisited and c[current].get(neighbor, 1) == 0:
                        S_tree[current].append(neighbor)
                        queue.append(neighbor)
                        unvisited.remove(neighbor)
                        if neighbor not in S_tree:
                            S_tree[neighbor] = []
                    elif  c[current].get(neighbor, 1) == 1:
                        T_tree[current] = T_tree.get(current, []) + [neighbor]
    
    return S_tree, T_tree

S_tree, T_tree = segment_apples(p, c)

##################################################################################
######################### AFISAREA IMAGINII SEGMENTATE ###########################
##################################################################################

def visualize_segmentation(img, S_tree, T_tree):
    segmented_img = img.copy()  

    for node, neighbors in S_tree.items():
        segmented_img[node[0], node[1]] = img[node[0], node[1]]  

        for neighbor in neighbors:
            segmented_img[neighbor[0], neighbor[1]] = img[neighbor[0], neighbor[1]] 

    for node, neighbors in T_tree.items():
        segmented_img[node[0], node[1]] = [0, 255, 0] 

        for neighbor in neighbors:
            segmented_img[neighbor[0], neighbor[1]] = [0, 255, 0]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_img)
    plt.title('Segmented Image')
    plt.axis('off')

    plt.show()

visualize_segmentation(img_brighter, S_tree, T_tree)


