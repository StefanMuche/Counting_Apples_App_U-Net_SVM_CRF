import numpy as np
from skimage import color
from matplotlib import pyplot as plt

def unary_potential(patch, label):
    avg_intensity = np.mean(patch)
    return 0 if (label == 0 and avg_intensity < 0.5) or (label == 1 and avg_intensity >= 0.5) else 1

def pairwise_potential(label1, label2):
    return 0 if label1 == label2 else 1

def gen_data_belief(image, labels):
    h, w = image.shape
    data_belief = np.zeros((h, w, len(labels)))
    
    for i in range(h):
        for j in range(w):
            patch = image[i:i+1, j:j+1]
            data_belief[i, j] = [unary_potential(patch, label) for label in labels]
    
    return data_belief

def loopy_belief_propagation(data_belief, labels, niter):
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

    return np.argmin(b, axis=2)

def demo_crf1(img):
    if len(img.shape) > 2:  
        img = color.rgb2gray(img)

    labels = [0, 1]
    data_belief = gen_data_belief(img, labels)
    segmented_image = loopy_belief_propagation(data_belief, labels, niter=2)
    plt.figure('Original Noisy Image')
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.title("Noisy Image")
    plt.figure('Segmented Image')
    plt.imshow(segmented_image, cmap='gray', interpolation='none')
    plt.title("Segmented Image")
    plt.show()
    return segmented_image

# Uncomment the following lines to run the demo with an example image
# from skimage.data import horse
# demo_crf(horse())
