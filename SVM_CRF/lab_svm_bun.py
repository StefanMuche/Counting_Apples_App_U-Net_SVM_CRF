import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, img_as_float, measure
from sklearn.preprocessing import scale
from joblib import load
import cv2
from pydensecrf import densecrf
from pydensecrf.utils import unary_from_labels, create_pairwise_gaussian, create_pairwise_bilateral

# File paths
image_path = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images_for_errors_png/resized_image_529.png'
model_path = 'D:/Python_VSCode/licenta_v2/Modules/svm_model_rbf_cielab_a_4clusters.joblib'
output_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/Labels_SVM+CRF_pydense' 

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

    return combined_features

def segment_image(image_path, model_path):
    image = img_as_float(io.imread(image_path))
    if image.shape[2] == 4:
        image = image[:, :, :3]
    features = extract_features(image)

    model = load(model_path)

    predicted_labels = model.predict(features)
    predicted_labels = predicted_labels.reshape(image.shape[:2])
   
    return predicted_labels

def retain_dark_gray_pixels(segmented_image):
    binary_mask = np.zeros_like(segmented_image)
    binary_mask[segmented_image == 2] = 1 
    output_image = binary_mask * 255

    return output_image

def apply_crf_to_binary_image(image, binary_mask):
    # Convert image to grayscale if it is not already
    if len(image.shape) > 2:
        image = color.rgb2gray(image)
    
    # Convert image and mask to the correct data types
    img = (image * 255).astype(np.uint8)
    binary_mask = (binary_mask / 255).astype(np.uint8)

    # Define the number of labels (2: background and foreground)
    num_labels = 2
    
    # Generate unary potentials
    labels = binary_mask
    unary = unary_from_labels(labels, num_labels, gt_prob=0.7, zero_unsure=False)

    # Create DenseCRF object
    d = densecrf.DenseCRF2D(img.shape[1], img.shape[0], num_labels)
    d.setUnaryEnergy(unary)

    # Add pairwise Gaussian potentials
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape)
    d.addPairwiseEnergy(feats, compat=3)

    # Add pairwise bilateral potentials
    img_3ch = np.stack([img]*3, axis=-1)
    feats = create_pairwise_bilateral(sdims=(3, 3), schan=(10,), img=img_3ch, chdim=2)
    d.addPairwiseEnergy(feats, compat=10)

    # Perform inference
    Q = d.inference(60)  # Increase number of iterations for better refinement
    map_result = np.argmax(Q, axis=0).reshape(img.shape)

    return map_result

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
    
    return num_apples, label_overlay, opening


def count_apples_svm(image_path, model_path):
    # Segment the image
    segmented_image = segment_image(image_path, model_path)
    result_image = retain_dark_gray_pixels(segmented_image)

    # Count apples and get labeled overlay image
    original_image = io.imread(image_path)
    num_apples, labeled_image = count_and_label_apples(result_image, original_image)
    
    return num_apples, labeled_image

# Load image and model
image = img_as_float(io.imread(image_path))
segmented_image = segment_image(image_path, model_path)
binary_mask = retain_dark_gray_pixels(segmented_image)

# Apply CRF refinement
crf_refined_mask = apply_crf_to_binary_image(image, binary_mask)

# Count apples in original and CRF-refined segmented images
num_apples_original, labeled_image_original, mata = count_and_label_apples(binary_mask, image)
num_apples_crf, labeled_image_crf, opening = count_and_label_apples(crf_refined_mask, image)
print(f"Numar mere SVM:{num_apples_original} ")
print(f"Numar mere SVM:{num_apples_crf} ")

# Display results
plt.figure('Original Image')
plt.imshow(image)
plt.title("Original Image")

plt.figure('Binary Mask')
plt.imshow(binary_mask, cmap='gray')
plt.title("Binary Mask")

plt.figure('Labeled Apples (Original Segmentation)')
plt.imshow(labeled_image_original)
plt.title(f"Labeled Apples (Original Segmentation) - Count: {num_apples_original}")

plt.figure('CRF Refined Mask')
plt.imshow(opening, cmap='gray')
plt.title("CRF Refined Mask")

plt.figure('Labeled Apples (CRF Segmentation)')
plt.imshow(labeled_image_crf)
plt.title(f"Labeled Apples (CRF Segmentation) - Count: {num_apples_crf}")

plt.show()
