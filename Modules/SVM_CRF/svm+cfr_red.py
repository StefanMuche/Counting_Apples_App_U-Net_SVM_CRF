import numpy as np
from skimage import io, img_as_float
from joblib import load
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, filters, img_as_float
from sklearn.preprocessing import scale
from joblib import dump
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

IMAGE_PATH = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images4'

# def verify_red(image):
#     image = img_as_float(image)
#     red_channel = image[:, :, 0]
#     green_channel = image[:, :, 1]
#     blue_channel = image[:, :, 2]
    
#     is_red = (red_channel > green_channel) & (red_channel > blue_channel) & (red_channel > 50 / 255.0)
#     red_features = np.where(is_red, 1, 0).flatten()
#     all_features = np.hstack([red_features])
#     return all_features

def verify_red(image):
    # Normalizează imaginea
    image = img_as_float(image)
    
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    
    # Condițiile pentru a fi considerat roșu
    is_red = (red_channel > green_channel) & (red_channel > blue_channel) & (red_channel > 50 / 255.0) & (green_channel > blue_channel)
    # Atribuie valoarea 1 dacă îndeplinește condițiile, altfel 0
    red_features = np.where(is_red, 1, 0).flatten()
    
    # Adăugăm o dimensiune suplimentară pentru a obține forma (921600, 1)
    red_features = red_features[:, np.newaxis]
    
    return red_features

def process_folder(folder_path):
    all_features_list = []
    for filename in tqdm(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, filename)
        image = io.imread(image_path)
        features = verify_red(image)
        all_features_list.append(features)
    
    # Convertim lista la un array numpy cu forma (num_imagini, 921600, 1)
    return np.array(all_features_list)

all_features = process_folder(IMAGE_PATH)
print(all_features.shape)  # Verificăm dimensiunile caracteristicilor
print("DONE PRE-PROCESSING IMAGES...")

def fuzzy_c_means_clustering(features, n_clusters=2, m=2.0):
    # Assuming 'features' is passed as a numpy array with shape (4, 921600, 19)
    # You need to reshape it to (number of features, number of samples)
    # Concatenate along the feature axis and reshape accordingly
    num_samples = features.shape[1] * features.shape[0]  # 921600
    num_features = features.shape[2]  # 4 * 19 = 76

    # Reshape to (76, 921600)
    features_normalized = features.transpose(1, 0, 2).reshape(num_samples, num_features).T

    cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
        features_normalized, c=n_clusters, m=m, error=0.005, maxiter=1000
    )
    return features_normalized, u_orig

def select_training_samples(memberships, nJ=600000):
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
    features = features.T  # Transpose features to have samples on rows
    labels = np.argmax(memberships, axis=0)
    print("Features:", features.shape)
    print("Labels:", labels.shape)
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    model = LinearSVC(max_iter=2000)
    model.fit(train_features, train_labels)
    return model

print("START FUZZY C-MEANS CLUSTERING...")
features_normalized, memberships = fuzzy_c_means_clustering(all_features)
print(features_normalized.shape)
print(memberships.shape)
print("DONE FUZZY C-MEANS CLUSTERING...")

print("START TRAINING SVM...")
train_indices, test_indices = select_training_samples(memberships)
print(features_normalized.shape)
print("Training SVM on selected samples...")
svm_model = train_svm_on_selected_samples(features_normalized, memberships, train_indices)

dump(svm_model, 'svm_model_red_threshold5.joblib')
print("DONE TRAINING SVM...")
