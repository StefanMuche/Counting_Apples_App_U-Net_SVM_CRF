import numpy as np
from skimage import io, color, filters, img_as_float
from sklearn.preprocessing import scale
import skfuzzy as fuzz
from sklearn.svm import LinearSVC, SVC
from skimage import graph
from skimage.segmentation import relabel_sequential
import os
from tqdm import tqdm
from joblib import dump, load
import matplotlib.pyplot as plt

IMAGE_PATH = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images4'
MASK_PATH = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/masks'

print("START PRE-PROCESSING IMAGES...")

def apply_gaussian_filter(image, sigma=2):
    """Apply Gaussian filter to the image to reduce noise."""
    return filters.gaussian(image, sigma=sigma)

def extract_features(image):

    filtered_image = apply_gaussian_filter(image, sigma=2)
    
    # Convert to CIELAB color space
    lab_image = color.rgb2lab(filtered_image)
    
    # Extract 'a' channel
    a_channel = lab_image[:, :, 1]

    normalized_a_channel = scale(a_channel.flatten()).reshape(a_channel.shape)
    combined_features = normalized_a_channel.reshape(-1, 1)

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
all_features = all_features.reshape(all_features.shape[0], -1, 1)
print(all_features.shape)

print("DONE PRE-PROCESSING IMAGES...")

def fuzzy_c_means_clustering(features, n_clusters=4, m=2.0):
    num_samples = features.shape[1] * features.shape[0]
    num_features = features.shape[2]

    features_normalized = features.transpose(1, 0, 2).reshape(num_samples, num_features).T

    cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
        features_normalized, c=n_clusters, m=m, error=0.005, maxiter=1000
    )
    return features_normalized, u_orig

def select_training_samples(memberships, nJ=2000):
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
    
    # model = LinearSVC(max_iter=2000)
    model = SVC(kernel='rbf')

    model.fit(train_features, train_labels)
    return model

print("START FUZZY C-MEANS CLUSTERING...")

features_normalized, memberships = fuzzy_c_means_clustering(all_features)
print("Visualizing clusters...")

print("DONE FUZZY C-MEANS CLUSTERING...")

print("START TRAINING SVM...")

train_indices, test_indices = select_training_samples(memberships)

print("Training SVM on selected samples...")
svm_model = train_svm_on_selected_samples(features_normalized, memberships, train_indices)

# Save the model
dump(svm_model, 'svm_model_rbf_cielab_a_4clusters.joblib')

print("DONE TRAINING SVM...")
