import numpy as np
from skimage import io, img_as_float, color, filters
from skimage.color import rgb2hsv
from sklearn.preprocessing import scale
from joblib import load
import matplotlib.pyplot as plt
from scipy.stats import skew, weibull_min
import os

image_path = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images_svm/20150921_131453_image1196.png'
model_path = 'D:/Python_VSCode/licenta_v2/svm_model_linear_gabor_HSV_2p.joblib'

def enhance_v_plane(v_plane):
    """Enhance the V plane using Weibull distribution."""
    mean_v = np.mean(v_plane)
    std_v = np.std(v_plane)
    skewness_v = skew(v_plane.flatten())
    
    if skewness_v != 0:
        shape, loc, scale_param = weibull_min.fit(v_plane.flatten(), floc=0)
        v_enhanced = weibull_min(shape, loc, scale_param).rvs(v_plane.shape)
        
        lv = np.mean(v_enhanced)
        v_transformed = np.where((v_enhanced < lv) & (skewness_v > 0), lv,
                                 np.where((v_enhanced > lv) & (skewness_v < 0), lv, v_enhanced))
        return v_transformed
    return v_plane

def extract_features(image, block_size=2):
    h, w, _ = image.shape
    h = (h // block_size) * block_size
    w = (w // block_size) * block_size
    image = image[:h, :w]

    hsv_image = rgb2hsv(image)
    v_plane = hsv_image[:, :, 2]
    v_plane_enhanced = enhance_v_plane(v_plane)

    hsv_image[:, :, 2] = v_plane_enhanced
    enhanced_image = color.hsv2rgb(hsv_image)

    ycbcr_image = color.rgb2ycbcr(enhanced_image)
    y_channel = ycbcr_image[:, :, 0]

    frequencies = [0.5, 1.0, 2.0]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    gabor_features = []

    for frequency in frequencies:
        for theta in thetas:
            real, imag = filters.gabor(y_channel, frequency=frequency, theta=theta)
            gabor_features.append(np.abs(real))

    gabor_features = np.stack(gabor_features, axis=-1)
    pixel_texture_feature = np.max(gabor_features, axis=-1)
    pixel_texture_feature = scale(pixel_texture_feature.flatten()).reshape(h, w)
    normalized_v_plane = scale(v_plane_enhanced.flatten()).reshape(h, w)

    combined_features = np.stack((pixel_texture_feature, normalized_v_plane), axis=-1)

    combined_features_blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = combined_features[i:i+block_size, j:j+block_size].reshape(-1)
            combined_features_blocks.append(block)
    combined_features_blocks = np.array(combined_features_blocks)

    return combined_features_blocks, enhanced_image

def segment_image(image_path, model_path, block_size=20):
    # Load the image
    image = img_as_float(io.imread(image_path))

    # Extract features from the image and get the enhanced RGB image after preprocessing
    features, enhanced_image = extract_features(image, block_size=block_size)
    
    # Load the trained SVM model
    model = load(model_path)
    
    # Predict the labels using the SVM model
    predicted_labels = model.predict(features)
    
    # Reshape predicted labels to match the image dimensions
    h, w, _ = image.shape
    h = (h // block_size) * block_size
    w = (w // block_size) * block_size
    predicted_labels = predicted_labels.reshape(h // block_size, w // block_size)
    predicted_labels = np.kron(predicted_labels, np.ones((block_size, block_size)))

    # Display the enhanced RGB image after preprocessing
    plt.figure(figsize=(10, 8))
    plt.imshow(enhanced_image)
    plt.title('Enhanced RGB Image after Preprocessing')
    plt.axis('off')
    plt.show()
    
    # Display the segmented image
    plt.figure(figsize=(10, 8))
    plt.imshow(predicted_labels, cmap='gray')
    plt.title('Segmented Image')
    plt.axis('off')
    plt.show()

segment_image(image_path, model_path)
