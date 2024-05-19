
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, filters, img_as_float
from sklearn.preprocessing import scale
from joblib import dump
from skimage.color import rgb2lab
from sklearn.preprocessing import scale
from joblib import dump, load
from scipy.stats import skew, weibull_min

image_path = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images_svm/20150921_131453_image636.png'
model_path = 'D:/Python_VSCode/licenta_v2/svm_model_linear_gabor_HSV.joblib'

def enhance_v_plane(v_plane):

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

def extract_features(image):

    hsv_image = color.rgb2hsv(image)
    # brightness_increase = 0.2
    # hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] + brightness_increase, 0, 1)
    v_plane = hsv_image[:, :, 2]  # Extract the V plane
    img = hsv_image
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
            gabor_features.append(np.abs(real).flatten())  


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

    return combined_features, enhanced_image


def segment_image(image_path, model_path):

    image = img_as_float(io.imread(image_path))

    # Extract features from the image and get the enhanced RGB image after preprocessing
    features, enhanced_image = extract_features(image)

    model = load(model_path)

    predicted_labels = model.predict(features)
    predicted_labels = predicted_labels.reshape(image.shape[:2])

    plt.figure(figsize=(10, 8))
    plt.imshow(enhanced_image)
    plt.title('Enhanced RGB Image after Preprocessing')
    plt.axis('off')
    plt.show()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(predicted_labels, cmap='gray')
    plt.title('Segmented Image')
    plt.axis('off')
    plt.show()


segment_image(image_path, model_path)

