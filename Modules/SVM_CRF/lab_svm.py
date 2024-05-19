import numpy as np
from skimage import io, color, filters, img_as_float
from skimage.color import rgb2lab
from sklearn.preprocessing import scale
import skfuzzy as fuzz
from sklearn.svm import SVC, LinearSVC
from skimage import graph
from skimage.segmentation import relabel_sequential
import os
from tqdm import tqdm
from joblib import dump, load
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter

IMAGE_PATH = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images4'
MASK_PATH = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/masks'

print("START PRE-PROCESSING IMAGES...")

def compute_local_homogeneity(lab_image, window_size=5):
    """Compute the local homogeneity for the LAB image."""
    half_window = window_size // 2

    def local_homogeneity(window):
        center = window[window_size**2 // 2]
        local_mean = np.mean(window)
        local_std = np.std(window)
        sobel_x = filters.sobel_h(window.reshape(window_size, window_size))
        sobel_y = filters.sobel_v(window.reshape(window_size, window_size))
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        local_discontinuity = np.mean(gradient_magnitude)
        
        # Prevent division by zero
        normalized_std = 0 if np.max(local_std) == 0 else local_std / np.max(local_std)
        normalized_discontinuity = 0 if np.max(local_discontinuity) == 0 else local_discontinuity / np.max(local_discontinuity)
        
        return 1 - normalized_discontinuity * normalized_std

    homogeneity_features = np.zeros_like(lab_image)
    for k in range(3):  # For each color component L, a, b
        homogeneity_features[:, :, k] = generic_filter(lab_image[:, :, k], local_homogeneity, size=window_size)

    return homogeneity_features

def extract_features(image):
    # Step 1: Transform the color image from RGB to LAB color space
    lab_image = color.rgb2lab(image)
    l_channel = lab_image[:, :, 0]  # Extract the L component
    a_channel = lab_image[:, :, 1]  # Extract the a component
    b_channel = lab_image[:, :, 2]  # Extract the b component

    # Normalize the LAB channels
    l_channel_normalized = scale(l_channel.flatten()).reshape(l_channel.shape)
    a_channel_normalized = scale(a_channel.flatten()).reshape(a_channel.shape)
    b_channel_normalized = scale(b_channel.flatten()).reshape(b_channel.shape)

    # Compute local homogeneity
    homogeneity_features = compute_local_homogeneity(lab_image)

    # Step 2: Apply the Gabor filter to the L channel
    frequencies = [0.5, 1.0, 2.0]  # Scale subbands
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Orientation bands
    gabor_features = []

    for frequency in frequencies:
        for theta in thetas:
            real, imag = filters.gabor(l_channel_normalized, frequency=frequency, theta=theta)
            gabor_features.append(np.abs(real).flatten())  # We use the magnitude of the response

    # Stack all Gabor features along the last axis
    gabor_features = np.stack(gabor_features, axis=-1)

    # Extract pixel texture feature
    # For each pixel, take the maximum absolute value across all Gabor features
    pixel_texture_feature = np.max(gabor_features, axis=-1)

    # Normalize the texture features
    pixel_texture_feature = scale(pixel_texture_feature.flatten()).reshape(pixel_texture_feature.shape)

    # Combine LAB, homogeneity, and Gabor features
    l_channel_normalized = l_channel_normalized.reshape(-1, 1)
    a_channel_normalized = a_channel_normalized.reshape(-1, 1)
    b_channel_normalized = b_channel_normalized.reshape(-1, 1)
    homogeneity_features = homogeneity_features.reshape(-1, 3)
    pixel_texture_feature = pixel_texture_feature.reshape(-1, 1)

    combined_features = np.hstack((l_channel_normalized, a_channel_normalized, b_channel_normalized, homogeneity_features, pixel_texture_feature))

    return combined_features

def process_folder(folder_path):
    all_features_list = []
    for filename in tqdm(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, filename)
        image = img_as_float(io.imread(image_path))
        features = extract_features(image)
        all_features_list.append(features)
    return np.array(all_features_list)

all_features = process_folder(IMAGE_PATH)
all_features = all_features.reshape(all_features.shape[0], -1, all_features.shape[-1])
print(all_features.shape)

print("DONE PRE-PROCESSING IMAGES...")

def fuzzy_c_means_clustering(features, n_clusters=2, m=2.0):
    num_samples = features.shape[1] * features.shape[0]
    num_features = features.shape[2]

    features_normalized = features.transpose(1, 0, 2).reshape(num_samples, num_features).T

    cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
        features_normalized, c=n_clusters, m=m, error=0.005, maxiter=1000
    )
    return features_normalized, u_orig

def select_training_samples(memberships, nJ=2000000):
    num_samples = memberships.shape[1]
    num_clusters = memberships.shape[0]
    train_indices = []
    test_indices = []

    for j in range(num_clusters):
        cluster_indices = np.argsort(-memberships[j])
        train_indices.extend(cluster_indices[:nJ])
        test_indices.extend(cluster_indices[nJ:])

    return train_indices, test_indices

def train_svm_on_selected_samples(features, memberships, train_indices):
    features = features.T
    labels = np.argmax(memberships, axis=0)
    print("Features:", features.shape)
    print("Labels:", labels.shape)
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    model = LinearSVC(max_iter=2000)
    model.fit(train_features, train_labels)
    return model

def apply_crf(image, labels):
    g = graph.pixel_graph(image, labels, mode='similarity')
    new_labels = graph.cut_normalized(labels, g)
    new_labels = relabel_sequential(new_labels)[0]
    return new_labels

print("START FUZZY C-MEANS CLUSTERING...")

features_normalized, memberships = fuzzy_c_means_clustering(all_features)
print("Visualizing clusters...")

print("DONE FUZZY C-MEANS CLUSTERING...")

print("START TRAINING SVM...")

train_indices, test_indices = select_training_samples(memberships)

print("Training SVM on selected samples...")
svm_model = train_svm_on_selected_samples(features_normalized, memberships, train_indices)

dump(svm_model, 'svm_model_linear_LAB_Gabor_Homogeneity.joblib')

print("DONE TRAINING SVM...")
