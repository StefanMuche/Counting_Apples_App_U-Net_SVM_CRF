import numpy as np
from skimage import io, color, filters, img_as_float
from skimage.color import rgb2lab
from sklearn.preprocessing import scale
from sklearn.svm import LinearSVC, SVC
from skimage import graph
from skimage.segmentation import relabel_sequential
import os
from tqdm import tqdm
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

IMAGE_PATH = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images4'
MASK_PATH = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/masks'

def extract_features(image):
    gray_image = color.rgb2gray(image)
    lab_image = rgb2lab(image)
    red_channel = image[:, :, 0]
    frequencies = [0.1, 0.3]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    gabor_features = []

    for frequency in frequencies:
        for theta in thetas:
            real, imag = filters.gabor(gray_image, frequency=frequency, theta=theta)
            gabor_features.append(real.flatten())
            gabor_features.append(imag.flatten())
    red_features = red_channel.flatten()
    gabor_features = np.stack(gabor_features, axis=-1)
    lab_features = lab_image.reshape((-1, 3))
    all_features = np.hstack([lab_features, gabor_features, red_features[:, np.newaxis]])

    return scale(all_features)

def process_folder(folder_path):
    all_features_list = []
    for filename in tqdm(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, filename)
        image = img_as_float(io.imread(image_path))
        features = extract_features(image)
        all_features_list.append(features)
    
    # Transformăm lista de matrici într-o singură matrice 2D
    all_features_array = np.vstack(all_features_list)
    
    return all_features_array

all_features = process_folder(IMAGE_PATH)

def k_means_clustering(features, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)  # Fără transpusă
    return kmeans.labels_

def select_training_samples(labels, nJ=600000):
    """
    Selects nJ samples from each cluster to form the training set.
    """
    num_samples = len(labels)
    num_clusters = np.unique(labels).size
    train_indices = []
    test_indices = []

    for j in range(num_clusters):
        cluster_indices = np.where(labels == j)[0]
        np.random.shuffle(cluster_indices)  # Shuffle indices for randomness
        train_indices.extend(cluster_indices[:nJ])  # Top nJ indices for training
        test_indices.extend(cluster_indices[nJ:])  # Remaining indices for testing

    return train_indices, test_indices

def train_svm_on_selected_samples(features, labels, train_indices):
    """
    Train the SVM using the selected samples.
    """
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    model = SVC(kernel = 'rbf')
    model.fit(train_features, train_labels)
    return model

print("START PRE-PROCESSING IMAGES...")

print("DONE PRE-PROCESSING IMAGES...")

print("START K-MEANS CLUSTERING...")

labels = k_means_clustering(all_features)

print("DONE K-MEANS CLUSTERING...")

print("START TRAINING SVM...")
train_indices, test_indices = select_training_samples(labels)

print("Training SVM on selected samples...")
svm_model = train_svm_on_selected_samples(all_features, labels, train_indices)

dump(svm_model, 'svm_model_linear_kmeans_4clusters.joblib')

print("DONE TRAINING SVM...")
