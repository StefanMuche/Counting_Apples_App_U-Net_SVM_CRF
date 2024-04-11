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
    if label == 0:  # Background
        return 0 if avg_intensity < 0.5 else 1
    else:  # Foreground
        return 1 if avg_intensity < 0.5 else 0


def pairwise_potential(label1, label2):
    return 0 if label1 == label2 else 1


def gen_data_belief(image, labels):
    h, w = image.shape
    patch_size = 4
    data_belief = np.zeros((h // patch_size, w // patch_size, len(labels)))
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            for k, label in enumerate(labels):
                pi, pj = i // patch_size, j // patch_size
                data_belief[pi, pj, k] = unary_potential(patch, label)
    return data_belief

# Loopy Belief Propagation
def loopy_belief_propagation(data_belief, labels, niter, patch_size=4):
    b = np.copy(data_belief)
    for iteration in range(niter):
        for i in range(data_belief.shape[0]):
            for j in range(data_belief.shape[1]):
                for k, label in enumerate(labels):
                    sum_pairwise = 0
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < data_belief.shape[0] and 0 <= nj < data_belief.shape[1]:
                            sum_pairwise += pairwise_potential(label, np.argmin(b[ni, nj]))
                    b[i, j, k] = data_belief[i, j, k] + sum_pairwise
        b -= np.min(b, axis=2, keepdims=True)
    return np.argmin(b, axis=2) * patch_size  

def demo_crf():
    TRAIN_IMAGES_PATH = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/test/predicted_masks'
    images = os.listdir(TRAIN_IMAGES_PATH)
    img_path = os.path.join(TRAIN_IMAGES_PATH, 'predicted_mask_316.jpg')
    img = io.imread(img_path)
    if len(img.shape) > 2:  
        img = color.rgb2gray(img)
    # img = np.array(horse(), dtype=bool)
    # noise = np.random.binomial(1, 0.02, img.shape)
    # noise = dilation(noise)
    # noisy_img = np.logical_xor(img, noise)

    labels = [0, 1]  # Background and Foreground
    data_belief = gen_data_belief(img, labels)

    start_time = clock()
    segmented_image = loopy_belief_propagation(data_belief, labels, niter=10)
    run_time = clock() - start_time

    plt.figure('Original Noisy Image')
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.title("Noisy Image")
    plt.figure('Segmented Image')
    plt.imshow(segmented_image, cmap='gray', interpolation='none')
    plt.title(f"Segmented Image - Runtime: {run_time:.2f}s")
    plt.show()

if __name__ == '__main__':
    demo_crf()
