import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, filters, img_as_float, measure
from sklearn.preprocessing import scale
from joblib import load
import cv2
from sklearn.metrics import accuracy_score, mean_squared_error

image_path = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images_for_errors_png'
mask_path = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/Masks_for_errors'
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

def calculate_metrics(segmented_image, original_mask):
    # Flatten the images
    segmented_flat = segmented_image.flatten()
    original_flat = original_mask.flatten()
    
    # Calculate accuracy
    accuracy = accuracy_score(original_flat, segmented_flat)
    
    # Calculate mean squared error (MSE)
    mse = mean_squared_error(original_flat, segmented_flat)
    
    return accuracy, mse

def compare_and_evaluate(image_path, mask_path, model_path):
    # Segment the image
    segmented_image = segment_image(image_path, model_path)
    result_image = retain_dark_gray_pixels(segmented_image)

    # Load the original mask
    original_mask = io.imread(mask_path)
    if original_mask.shape[2] == 4:
        original_mask = original_mask[:, :, :3]
    if len(original_mask.shape) == 3:
        original_mask = color.rgb2gray(original_mask) * 255
    original_mask = original_mask.astype(np.uint8)

    # Calculate accuracy and MSE
    accuracy, mse = calculate_metrics(result_image, original_mask)

    print(f'Accuracy: {accuracy}')
    print(f'Mean Squared Error: {mse}')

    # Display results
    plt.figure(figsize=(10, 8))
    plt.imshow(segmented_image, cmap='gray')
    plt.title('Segmented Image')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(result_image, cmap='gray')
    plt.title('Retained Dark Gray Pixels')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(original_mask, cmap='gray')
    plt.title('Original Mask')
    plt.axis('off')
    plt.show()

compare_and_evaluate(image_path, mask_path, model_path)
