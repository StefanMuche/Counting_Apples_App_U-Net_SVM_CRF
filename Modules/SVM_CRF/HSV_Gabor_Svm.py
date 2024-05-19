import numpy as np
from skimage import io, color, filters, img_as_float
from skimage.color import rgb2lab, rgb2hsv
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
from skimage.filters import sobel
from scipy.ndimage import generic_filter
from scipy.stats import skew, weibull_min

IMAGE_PATH = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images4'
MASK_PATH = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/masks'

print("START PRE-PROCESSING IMAGES...")

def enhance_v_plane(v_plane):
    """Enhance the V plane using Weibull distribution."""
    mean_v = np.mean(v_plane)
    std_v = np.std(v_plane)
    skewness_v = skew(v_plane.flatten())
    
    # Check if enhancement is needed
    if skewness_v != 0:
        shape, loc, scale_param = weibull_min.fit(v_plane.flatten(), floc=0)
        v_enhanced = weibull_min(shape, loc, scale_param).rvs(v_plane.shape)
        
        lv = np.mean(v_enhanced)
        v_transformed = np.where((v_enhanced < lv) & (skewness_v > 0), lv,
                                 np.where((v_enhanced > lv) & (skewness_v < 0), lv, v_enhanced))
        return v_transformed
    return v_plane

def extract_features(image):

    hsv_image = color.rgb2hsv(image)
    # brightness_increase = 0.2
    # hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] + brightness_increase, 0, 1)
    v_plane = hsv_image[:, :, 2]  # Extract the V plane

    # Enhance the V plane
    v_plane_enhanced = enhance_v_plane(v_plane)

   
    hsv_image[:, :, 2] = v_plane_enhanced
    enhanced_image = color.hsv2rgb(hsv_image)

    ycbcr_image = color.rgb2ycbcr(enhanced_image)
    y_channel = ycbcr_image[:, :, 0]  # Extract the luminance component Y

  
    frequencies = [0.5, 1.0, 2.0]  # Scale subbands
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Orientation bands
    gabor_features = []

    for frequency in frequencies:
        for theta in thetas:
            real, imag = filters.gabor(y_channel, frequency=frequency, theta=theta)
            gabor_features.append(np.abs(real).flatten())  # We use the magnitude of the response

  
    gabor_features = np.stack(gabor_features, axis=-1)

   
    # For each pixel, take the maximum absolute value across all Gabor features
    pixel_texture_feature = np.max(gabor_features, axis=-1)

    # Normalize the texture features
    pixel_texture_feature = scale(pixel_texture_feature.flatten()).reshape(pixel_texture_feature.shape)

    # Normalize the enhanced V plane
    normalized_v_plane = scale(v_plane_enhanced.flatten()).reshape(v_plane_enhanced.shape)

    # Combine texture and normalized V plane features
    pixel_texture_feature = pixel_texture_feature.reshape(-1, 1)
    normalized_v_plane = normalized_v_plane.reshape(-1, 1)
    combined_features = np.hstack((pixel_texture_feature, normalized_v_plane))

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


def fuzzy_c_means_clustering(features, n_clusters=2, m=2.0):
    
    num_samples = features.shape[1] * features.shape[0] 
    num_features = features.shape[2] 

   
    features_normalized = features.transpose(1, 0, 2).reshape(num_samples, num_features).T

    cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
        features_normalized, c=n_clusters, m=m, error=0.005, maxiter=1000
    )
    return features_normalized, u_orig

def select_training_samples(memberships, nJ=2000000):
    """
    Selects nJ samples with the highest membership values from each cluster to form the training set.
    """
    num_samples = memberships.shape[1]
    num_clusters = memberships.shape[0]
    train_indices = []
    test_indices = []

    for j in range(num_clusters):
        cluster_indices = np.argsort(-memberships[j])  # Sort indices by membership value, descending
        train_indices.extend(cluster_indices[:nJ])  # Top nJ indices for training
        test_indices.extend(cluster_indices[nJ:])  # Remaining indices for testing

    return train_indices, test_indices

def train_svm_on_selected_samples(features, memberships, train_indices):
    """
    Train the SVM using the selected samples.
    """
    features = features.T  # Transpose features to have samples on rows
    labels = np.argmax(memberships, axis=0)
    print("Features:", features.shape)
    print("Labels:", labels.shape)
    # Select training samples and labels
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    model = LinearSVC(max_iter= 2000)
    # model = SVC(kernel='poly')
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
# plot_clusters(features_normalized, memberships)

print("DONE FUZZY C-MEANS CLUSTERING...")

print("START TRAINING SVM...")

train_indices, test_indices = select_training_samples(memberships)

print("Training SVM on selected samples...")
svm_model = train_svm_on_selected_samples(features_normalized, memberships, train_indices)

#svm_model = train_svm(features_normalized, memberships)

# Salvarea modelului
dump(svm_model, 'svm_model_linear_gabor_HSV.joblib')

# Încărcarea modelului
# svm_model = load('svm_model.joblib')

print("DONE TRAINING SVM...")



