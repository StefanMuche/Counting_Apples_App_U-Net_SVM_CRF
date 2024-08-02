import numpy as np
from scipy import ndimage as ndi
from skimage.transform import rescale, resize
from skimage.data import horse
from skimage.morphology import dilation
from matplotlib import pyplot as plt
from time import process_time as clock
import os
from skimage import io, color


def unary_potential(patch, label):
    avg_intensity = np.mean(patch)
    return 0 if (label == 0 and avg_intensity < 0.5) or (label == 1 and avg_intensity >= 0.5) else 1

def pairwise_potential(label1, label2):
    return 0 if label1 == label2 else 1

def gen_data_belief(image, labels):
    h, w = image.shape
    patch_size = 4
    data_belief = np.zeros((h // patch_size, w // patch_size, len(labels)))
    
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            pi, pj = i // patch_size, j // patch_size
            data_belief[pi, pj] = [unary_potential(patch, label) for label in labels]
    
    return data_belief

def loopy_belief_propagation(data_belief, labels, niter, patch_size=1):
    b = np.copy(data_belief)
    h, w = data_belief.shape[:2]

    for iteration in range(niter):
        new_b = np.copy(b)
        
        for i in range(h):
            for j in range(w):
                for k, label in enumerate(labels):
                    sum_pairwise = 0
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            sum_pairwise += pairwise_potential(label, np.argmin(b[ni, nj]))
                    new_b[i, j, k] = data_belief[i, j, k] + sum_pairwise
        
        new_b -= np.min(new_b, axis=2, keepdims=True)
        b = new_b

    return np.argmin(b, axis=2) * patch_size  

def demo_crf(img):
    if len(img.shape) > 2:  
        img = color.rgb2gray(img)

    labels = [0, 1]
    data_belief = gen_data_belief(img, labels)
    segmented_image = loopy_belief_propagation(data_belief, labels, niter=5)
    plt.figure('Original Noisy Image')
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.title("Noisy Image")
    plt.figure('Segmented Image')
    plt.imshow(segmented_image, cmap='gray', interpolation='none')
    plt.show()
    return segmented_image
    # run_time = clock() - start_time



# if __name__ == '__main__':
#     demo_crf()
