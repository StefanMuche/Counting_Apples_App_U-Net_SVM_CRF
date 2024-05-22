import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, filters, img_as_float, measure
from sklearn.preprocessing import scale
from joblib import load
import cv2

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
    if image.shape[2] == 4:
        image = image[:, :, :3]
    features, filtered_image = extract_features(image)

    model = load(model_path)

    predicted_labels = model.predict(features)
    predicted_labels = predicted_labels.reshape(image.shape[:2])
   
    return predicted_labels

def retain_dark_gray_pixels(segmented_image):
    binary_mask = np.zeros_like(segmented_image)
    binary_mask[segmented_image == 2] = 1 
    output_image = binary_mask * 255

    return output_image

def count_and_label_apples(binary_image, original_image):
    # Apply thresholding
    ret, thresh = cv2.threshold(binary_image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    
    # Apply morphological opening
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    
    # Label the connected components
    labels = measure.label(opening, background=0)
    
    # Count all unique labels
    unique_labels = np.unique(labels)
    num_apples = len(unique_labels) - 1  # Exclude the background label
    
    # Display labels on the original image
    label_overlay = color.label2rgb(labels, image=original_image, bg_label=0, alpha=0.3)
    
    return num_apples, label_overlay

def count_apples_svm(image_path, model_path):

    # Segment the image
    segmented_image = segment_image(image_path, model_path)
    result_image = retain_dark_gray_pixels(segmented_image)

    # Count apples and get labeled overlay image
    original_image = io.imread(image_path)
    num_apples, labeled_image = count_and_label_apples(result_image, original_image)
    
    return num_apples

