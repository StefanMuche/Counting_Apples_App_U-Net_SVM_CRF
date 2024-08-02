
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, filters, img_as_float
from sklearn.preprocessing import scale
from joblib import dump
from skimage.color import rgb2lab
from sklearn.preprocessing import scale
from joblib import dump, load
from scipy.stats import skew, weibull_min

image_path = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images_svm/20150921_131453_image391.png'
model_path = 'D:/Python_VSCode/licenta_v2/Modules/svm_model_rbf_cielab_a_4clusters.joblib'

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

    return combined_features, filtered_image


def segment_image(image_path, model_path):

    image = img_as_float(io.imread(image_path))
    features, filtered_image = extract_features(image)

    model = load(model_path)

    predicted_labels = model.predict(features)
    predicted_labels = predicted_labels.reshape(image.shape[:2])

    plt.figure(figsize=(10, 8))
    plt.imshow(filtered_image)
    plt.title('Enhanced RGB Image after Preprocessing')
    plt.axis('off')
    plt.show()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(predicted_labels, cmap='gray_r')
    plt.title('Segmented Image')
    plt.axis('off')
    plt.show()


segment_image(image_path, model_path)

