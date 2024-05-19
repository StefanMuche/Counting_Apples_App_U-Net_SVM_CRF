import numpy as np
from skimage import io, color, filters, img_as_float
from skimage.color import rgb2lab, rgb2hsv
from sklearn.preprocessing import scale
import skfuzzy as fuzz
from sklearn.svm import SVC
from skimage import graph
from skimage.segmentation import relabel_sequential
import os
from tqdm import tqdm
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import skew, weibull_min

# Path to the image and model
image_path = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images_svm/20150921_132038_image271.png'  # Update this path to the image you want to segment
model_path = 'D:/Python_VSCode/licenta_v2/svm_model_linear_HSV_gabor_V_modificat.joblib'  # Path to the saved model

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

# def extract_features(image):
#     gray_image = color.rgb2gray(image)
#     hsv_image = rgb2hsv(image)

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

#     hsv_img_brighter = hsv_image.copy()
#     hsv_img_brighter[:,:,2] = np.clip(hsv_image[:,:,2] + brightness_increase, 0, 1)
#     img_brighter = color.hsv2rgb(hsv_img_brighter)
#     hsv_features = img_brighter.reshape((-1, 3))
#     all_features = np.hstack([hsv_features, gabor_features])

#     return scale(all_features)

def segment_image(image_path, model_path):
    # Load the image
    image = img_as_float(io.imread(image_path))

    # Extract features from the image
    features = extract_features(image)
    
    # Load the trained SVM model
    model = load(model_path)
    
    # Predict the labels using the SVM model
    predicted_labels = model.predict(features)
    print(predicted_labels.shape)
    predicted_labels = predicted_labels.reshape(image.shape[:2])
    
    # Display the segmented image
    plt.figure(figsize=(10, 8))
    plt.imshow(predicted_labels, cmap='gray_r')
    plt.title('Segmented Image')
    plt.axis('off')
    plt.show()


# Perform segmentation and display the image
segment_image(image_path, model_path)

# def extract_features(image):
#     gray_image = color.rgb2gray(image)
#     lab_image = rgb2lab(image)

#     frequencies = [0.1, 0.3]
#     thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
#     gabor_features = []

#     for frequency in frequencies:
#         for theta in thetas:
#             real, imag = filters.gabor(gray_image, frequency=frequency, theta=theta)
#             gabor_features.append(real.flatten())
#             gabor_features.append(imag.flatten())

#     gabor_features = np.stack(gabor_features, axis=-1)
#     lab_features = lab_image.reshape((-1, 3))
#     all_features = np.hstack([lab_features, gabor_features])

#     return scale(all_features)

# def extract_features(image):
#     # Convert to grayscale
#     gray_image = color.rgb2gray(image)
    
#     # Convert to HSV color space and adjust brightness
#     hsv_image = color.rgb2hsv(image)
#     brightness_increase = 0.8
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

# def extract_features(image):
#     # Flatten the RGB image to a 2D array (pixels x channels)
#     rgb_features = image.reshape((-1, 3))
    
#     # Optionally scale the features to standardize them
#     scaled_rgb_features = scale(rgb_features)

#     return scaled_rgb_features

# def extract_features(image):
#     gray_image = color.rgb2gray(image)
#     hsv_image = rgb2hsv(image)
#     brightness_increase = 0.5
#     hsv_img_brighter = hsv_image.copy()
#     hsv_img_brighter[:,:,2] = np.clip(hsv_image[:,:,2] + brightness_increase, 0, 1)

#     frequencies = [0.1, 0.3]
#     thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
#     gabor_features = []

#     for frequency in frequencies:
#         for theta in thetas:
#             real, imag = filters.gabor(gray_image, frequency=frequency, theta=theta)
#             gabor_features.append(real.flatten())
#             gabor_features.append(imag.flatten())

#     gabor_features = np.stack(gabor_features, axis=-1)
#     hsv_features = hsv_image.reshape((-1, 3))
#     all_features = np.hstack([hsv_features, gabor_features])

#     return scale(all_features)




