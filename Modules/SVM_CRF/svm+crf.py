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
from scipy.stats import skew, weibull_min

IMAGE_PATH = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images4'
MASK_PATH = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/masks'

def apply_pca(features, n_components=2):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features.T)  # Transpose because PCA expects samples in rows
    return principal_components


def plot_clusters(features, memberships):
    # Apply PCA
    principal_components = apply_pca(features)
    
    # Get the cluster assignments
    cluster_assignments = np.argmax(memberships, axis=0)

    # Plotting
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=cluster_assignments, cmap='viridis', alpha=0.5)
    plt.title('Cluster distribution after Fuzzy C-Means')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter)
    plt.show()

def verify_red(image):
    red_channel = image[:,:,0]
    red_features = red_channel.flatten()

    return red_features

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
    gray_image = color.rgb2gray(image)
    hsv_image = color.rgb2hsv(image)
    # brightness_increase = 0.2
    # hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] + brightness_increase, 0, 1)
    v_plane = hsv_image[:, :, 2]  # Extract the V plane

    # Enhance the V plane
    v_plane_enhanced = enhance_v_plane(v_plane)
    hsv_image[:, :, 2] = v_plane_enhanced
    hsv_features = hsv_image.reshape((-1, 3))
    frequencies = [0.1, 0.3]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    gabor_features = []

    for frequency in frequencies:
        for theta in thetas:
            real, imag = filters.gabor(gray_image, frequency=frequency, theta=theta)
            gabor_features.append(real.flatten())
            gabor_features.append(imag.flatten())
    gabor_features = np.stack(gabor_features, axis=-1)
    all_features = np.hstack([gabor_features, hsv_features])

    return scale(all_features)


def process_folder(folder_path):
    all_features_list = []
    for filename in tqdm(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, filename)
        image = img_as_float(io.imread(image_path))
        features = extract_features(image)
        all_features_list.append(features)
    return np.array(all_features_list)

all_features = process_folder(IMAGE_PATH)
print(all_features.shape)
# np.save('D:/Python_VSCode/licenta_v2/Numpy/features_svm_training.npy', all_features)

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

# def train_svm(features, memberships):
#     labels = np.argmax(memberships, axis=0)
#     print("Features:", features.shape)
#     print("Labels:", labels.shape)
#     features = features.T   
#     model = SVC(kernel='linear')
#     model.fit(features, labels)
#     return model

def select_training_samples(memberships, nJ=600000):
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
    new_labels = relabel_sequential(new_labels)[0]  # Relabel for consistency
    return new_labels

print("START FUZZY C-MEANS CLUSTERING...")

features_normalized, memberships = fuzzy_c_means_clustering(all_features)
print("Visualizing clusters...")
# plot_clusters(features_normalized, memberships)

print("DONE FUZZY C-MEANS CLUSTERING...")

print("START TRAINING SVM...")
# Assuming 'features_normalized' and 'memberships' are obtained from fuzzy_c_means_clustering
train_indices, test_indices = select_training_samples(memberships)

print("Training SVM on selected samples...")
svm_model = train_svm_on_selected_samples(features_normalized, memberships, train_indices)

#svm_model = train_svm(features_normalized, memberships)

# Salvarea modelului
dump(svm_model, 'svm_model_linear_HSV_gabor_V_modificat.joblib')

# Încărcarea modelului
# svm_model = load('svm_model.joblib')

print("DONE TRAINING SVM...")

# def extract_features(image):
#     # Flatten the RGB image to a 2D array (pixels x channels)
#     rgb_features = image.reshape((-1, 3))
    
#     # Optionally scale the features to standardize them
#     scaled_rgb_features = scale(rgb_features)

#     return scaled_rgb_features

# def extract_features(image):
#     # Convert to grayscale
#     gray_image = color.rgb2gray(image)
    
#     # Convert to HSV color space and adjust brightness
#     hsv_image = color.rgb2hsv(image)
#     brightness_increase = 0.5
#     hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] + brightness_increase, 0, 1)
    
#     # Extract the red channel from the original RGB image
#     red_channel = image[:, :, 0]  # Assuming image is in RGB format

#     # Parameters for Gabor filters
#     frequencies = [0.1, 0.3]
#     thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
#     gabor_features = []

#     # Apply Gabor filter with different frequencies and orientations
#     for frequency in frequencies:
#         for theta in thetas:
#             real, imag = filters.gabor(gray_image, frequency=frequency, theta=theta)
#             gabor_features.append(real.flatten())
#             gabor_features.append(imag.flatten())

#     # Stack all Gabor features
#     gabor_features = np.stack(gabor_features, axis=-1)
    
#     # Reshape HSV features to be a 2D array
#     hsv_features = hsv_image.reshape((-1, 3))
    
#     # Flatten the red channel to be a 2D array
#     red_features = red_channel.flatten()
    
#     # Concatenate HSV, Gabor, and Red channel features
#     all_features = np.hstack([hsv_features, gabor_features, red_features[:, np.newaxis]])

#     # Scale features for uniformity
#     return scale(all_features)

# 

# def extract_features(image):
#     gray_image = color.rgb2gray(image)
#     lab_image = rgb2hsv(image)

#     frequencies = [0.1, 0.3]
#     thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
#     gabor_features = []

#     for frequency in frequencies:
#         for theta in thetas:
#             real, imag = filters.gabor(gray_image, frequency=frequency, theta=theta)
#             gabor_features.append(real.flatten())
#             gabor_features.append(imag.flatten())

#     gabor_features = np.stack(gabor_features, axis=-1)
#     brightness_increase = 0.5

#     hsv_img_brighter = lab_image.copy()
#     hsv_img_brighter[:,:,2] = np.clip(lab_image[:,:,2] + brightness_increase, 0, 1)
#     img_brighter = color.hsv2rgb(hsv_img_brighter)
#     rgb_features = img_brighter.reshape((-1, 3))
#     all_features = np.hstack([rgb_features, gabor_features])

#     return scale(all_features)


