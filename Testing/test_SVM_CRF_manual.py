import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, img_as_float, measure
from skimage.transform import resize
from sklearn.preprocessing import scale
from joblib import load
import cv2
from pydensecrf import densecrf
from pydensecrf.utils import unary_from_labels, create_pairwise_gaussian, create_pairwise_bilateral
import os
from CRF_pathces import demo_crf
import tqdm
from crf_fara_patch import demo_crf1
# File paths
image_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images_for_errors_png'
model_path = 'D:/Python_VSCode/licenta_v2/Modules/svm_model_rbf_cielab_a_4clusters.joblib'
output_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/Labels_SVM+CRF_pydense'
output_file_path = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/apple_counts_crf_manual.txt'

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
    feats = create_pairwise_bilateral(sdims=(2, 2), schan=(10,), img=img_3ch, chdim=2)
    d.addPairwiseEnergy(feats, compat=10)

    # Perform inference
    Q = d.inference(10)  # Increase number of iterations for better refinement
    map_result = np.argmax(Q, axis=0).reshape(img.shape)

    return map_result

def count_and_label_apples(binary_image, original_image):
    # Resize binary_image to match the shape of original_image
    if binary_image.shape != original_image.shape[:2]:
        binary_image = resize(binary_image, original_image.shape[:2], anti_aliasing=True)

    # Apply thresholding
    # ret, thresh = cv2.threshold(binary_image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # kernel = np.ones((3, 3), np.uint8)
    
    # # Apply morphological opening
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    
    # Label the connected components
    labels = measure.label(binary_image, background=0)
    
    # Count all unique labels
    unique_labels = np.unique(labels)
    num_apples = len(unique_labels) - 1  # Exclude the background label
    
    # Display labels on the original image
    label_overlay = color.label2rgb(labels, image=original_image, bg_label=0, alpha=0.3)
    plt.figure('Labeled Apples (Original Segmentation)')
    plt.imshow(label_overlay)
    plt.show()
    
    return num_apples

def process_images(image_dir, model_path, output_dir, output_file_path):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file_path, 'w') as output_file:
        # Iterate through all images in the directory
        for filename in tqdm.tqdm(os.listdir(image_dir)):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(image_dir, filename)
                
                # Load image and model
                image = img_as_float(io.imread(image_path))
                segmented_image = segment_image(image_path, model_path)
                binary_mask = retain_dark_gray_pixels(segmented_image)

                # Apply CRF refinement
                crf_refined_mask = demo_crf(binary_mask)

                # Count apples in CRF-refined segmented image
                num_apples_crf = count_and_label_apples(crf_refined_mask, image)
                
                # Save the labeled image
                # plt.imsave(os.path.join(output_dir, filename), labeled_image_crf)
                
                # Write the number of apples to the output file
                output_file.write(f'{num_apples_crf}\n')
                output_file.flush()  # Ensure the output is written to the file in real-time
                
                # Print the number of apples
                print(f'{filename}: {num_apples_crf}')

# Run the processing function
process_images(image_dir, model_path, output_dir, output_file_path)
