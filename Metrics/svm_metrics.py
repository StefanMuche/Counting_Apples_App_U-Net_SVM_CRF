import os
import numpy as np
from tqdm import tqdm
import cv2
from skimage import io, color, filters, img_as_float, measure
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from joblib import load
from pydensecrf import densecrf
from pydensecrf.utils import unary_from_labels, create_pairwise_gaussian, create_pairwise_bilateral
from PIL import Image as PILImage

# Directories
images_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images_for_errors_png'
masks_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/Masks_for_errors'
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
    
    # Resize image to 768x1024
    image = cv2.resize(image, (768, 1024))

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

def apply_crf_to_binary_image(image_path, binary_mask):
    # Convert image to grayscale if it is not already
    image = img_as_float(io.imread(image_path))
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
    feats = create_pairwise_bilateral(sdims=(4, 4), schan=(10,), img=img_3ch, chdim=2)
    d.addPairwiseEnergy(feats, compat=10)

    # Perform inference
    Q = d.inference(20)  # Increase number of iterations for better refinement
    map_result = np.argmax(Q, axis=0).reshape(img.shape)

    return map_result

confusion_mats_svm = []
confusion_mats_crf = []

for image_name in tqdm(os.listdir(images_dir)):
    image_path = os.path.join(images_dir, image_name)
    mask_path = os.path.join(masks_dir, image_name)

    if not os.path.exists(mask_path):
        continue

    true_mask = np.array(PILImage.open(mask_path).convert('L')).astype(np.uint8)
    true_mask = cv2.resize(true_mask, (768, 1024))

    # Segment using SVM
    segmented_image = segment_image(image_path, model_path)
    binary_mask_svm = retain_dark_gray_pixels(segmented_image)

    # Refine using CRF
    refined_mask = apply_crf_to_binary_image(image_path, binary_mask_svm)

    # Calculate confusion matrices
    conf_mat_svm = confusion_matrix(true_mask.flatten(), binary_mask_svm.flatten(), labels=[0, 255])
    conf_mat_crf = confusion_matrix(true_mask.flatten(), refined_mask.flatten(), labels=[0, 255])

    # Normalize confusion matrices
    conf_mat_svm = conf_mat_svm / true_mask.size
    conf_mat_crf = conf_mat_crf / true_mask.size

    confusion_mats_svm.append(conf_mat_svm)
    confusion_mats_crf.append(conf_mat_crf)

# Calculate mean confusion matrices
total_confusion_svm = np.mean(np.array(confusion_mats_svm), axis=0)
total_confusion_crf = np.mean(np.array(confusion_mats_crf), axis=0)

print("Average Confusion Matrix for SVM:")
print(total_confusion_svm)

print("Average Confusion Matrix for SVM + CRF:")
print(total_confusion_crf)
