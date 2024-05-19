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
from sklearn.svm import SVC
from skimage import graph
from skimage.segmentation import relabel_sequential
import os
from tqdm import tqdm
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import skew, weibull_min
image_path = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images_svm/20150921_132038_image271.png'
model_path = 'D:/Python_VSCode/licenta_v2/svm_model_linear_LAB_Gabor.joblib'

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
    # Step 1: Transform the color image from RGB to HSV color space
    hsv_image = color.rgb2hsv(image)
    v_plane = hsv_image[:, :, 2]  # Extract the V plane

    # Enhance the V plane
    v_plane_enhanced = enhance_v_plane(v_plane)

    # Replace the V plane in the HSV image and convert back to RGB
    hsv_image[:, :, 2] = v_plane_enhanced
    enhanced_image = color.hsv2rgb(hsv_image)

    # Step 2: Transform the enhanced color image from RGB to LAB color space
    lab_image = color.rgb2lab(enhanced_image)
    l_channel = lab_image[:, :, 0]  # Extract the L component
    a_channel = lab_image[:, :, 1]  # Extract the a component
    b_channel = lab_image[:, :, 2]  # Extract the b component

    # Normalize the LAB channels
    l_channel_normalized = scale(l_channel.flatten()).reshape(l_channel.shape)
    a_channel_normalized = scale(a_channel.flatten()).reshape(a_channel.shape)
    b_channel_normalized = scale(b_channel.flatten()).reshape(b_channel.shape)

    # Step 3: Apply the Gabor filter to the L channel
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

    # Combine LAB and Gabor features
    l_channel_normalized = l_channel_normalized.reshape(-1, 1)
    a_channel_normalized = a_channel_normalized.reshape(-1, 1)
    b_channel_normalized = b_channel_normalized.reshape(-1, 1)
    pixel_texture_feature = pixel_texture_feature.reshape(-1, 1)

    combined_features = np.hstack((l_channel_normalized, a_channel_normalized, b_channel_normalized, pixel_texture_feature))

    return combined_features, enhanced_image



def segment_image(image_path, model_path):
    # Load the image
    image = img_as_float(io.imread(image_path))

    # Extract features from the image and get the enhanced RGB image after preprocessing
    features, enhanced_image = extract_features(image)
    
    # Load the trained SVM model
    model = load(model_path)
    
    # Predict the labels using the SVM model
    predicted_labels = model.predict(features)
    predicted_labels = predicted_labels.reshape(image.shape[:2])

    # Display the enhanced RGB image after preprocessing
    plt.figure(figsize=(10, 8))
    plt.imshow(enhanced_image)
    plt.title('Enhanced RGB Image after Preprocessing')
    plt.axis('off')
    plt.show()
    
    
    # Display the segmented image
    plt.figure(figsize=(10, 8))
    plt.imshow(predicted_labels, cmap='gray_r')
    plt.title('Segmented Image')
    plt.axis('off')
    plt.show()

    # Perform segmentation and display the image
segment_image(image_path, model_path)

