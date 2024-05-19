import numpy as np
from skimage import io, color, filters, img_as_float
from skimage.color import rgb2lab
from sklearn.preprocessing import scale
import skfuzzy as fuzz
from sklearn.svm import SVC
from skimage.future import graph
from skimage.segmentation import relabel_sequential
import os
from tqdm import tqdm

IMAGE_PATH = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images'
MASK_PATH = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/masks'

print("START PRE-PROCESSING IMAGES...")

def extract_features(image):
    gray_image = color.rgb2gray(image)
    lab_image = rgb2lab(image)

    frequencies = [0.1, 0.3]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    gabor_features = []

    for frequency in frequencies:
        for theta in thetas:
            real, imag = filters.gabor(gray_image, frequency=frequency, theta=theta)
            gabor_features.append(real.flatten())
            gabor_features.append(imag.flatten())

    gabor_features = np.stack(gabor_features, axis=-1)
    lab_features = lab_image.reshape((-1, 3))
    all_features = np.hstack([lab_features, gabor_features.reshape((-1, gabor_features.shape[2]))])

    return scale(all_features)

def process_folder(folder_path):
    all_features_list = []
    for filename in tqdm(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, filename)
        image = img_as_float(io.imread(image_path))
        features = extract_features(image)
        all_features_list.append(features)
    return all_features_list

all_features = process_folder(IMAGE_PATH)

print("DONE PRE-PROCESSING IMAGES...")

def fuzzy_c_means_clustering(features, n_clusters=2, m=2.0):
    # Normalizarea datelor
    features_normalized = scale(features)
    cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
        features_normalized.T, c=n_clusters, m=m, error=0.005, maxiter=1000
    )
    return u_orig

def train_svm(features, memberships):
    labels = np.argmax(memberships, axis=0)
    model = SVC(kernel='linear')
    model.fit(features, labels)
    return model

def apply_crf(image, labels):
    g = graph.pixel_graph(image, labels, mode='similarity')
    new_labels = graph.cut_normalized(labels, g)
    new_labels = relabel_sequential(new_labels)[0]  # Relabel for consistency
    return new_labels

print("START FUZZY C-MEANS CLUSTERING...")

memberships = fuzzy_c_means_clustering(all_features)

print("DONE FUZZY C-MEANS CLUSTERING...")

print("START TRAINING SVM...")

svm_model = train_svm(all_features, memberships)

print("DONE TRAINING SVM...")

# def segment_image(image_path):
#     features = extract_features(image_path)
#     memberships = fuzzy_c_means_clustering(features)
#     svm_model = train_svm(features, memberships)
#     # initial_labels = svm_model.predict(features).reshape(image.shape[:2])
#     # refined_labels = apply_crf(image, initial_labels)
    
#     return refined_labels

# # Aplicarea pe o imagine
# image_path = 'path_to_apple_image.jpg'
# segmented_image = segment_image(image_path)
# io.imshow(segmented_image)
# io.show()

