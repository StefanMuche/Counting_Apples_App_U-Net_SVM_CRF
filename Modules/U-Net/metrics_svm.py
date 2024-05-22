import os
import numpy as np
from tqdm import tqdm
import cv2
from skimage import io, color, filters, img_as_float, measure
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score, mean_squared_error
from joblib import load
import torch.nn.functional as F
import torch

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

def calculate_metrics(segmented_image, original_mask):
    # Flatten the images
    segmented_flat = segmented_image.flatten()
    original_flat = original_mask.flatten()

    # Calculate accuracy
    accuracy = accuracy_score(original_flat, segmented_flat)
    
    # Calculate mean squared error (MSE)
    mse = mean_squared_error(original_flat, segmented_flat)

    # Convert arrays to tensors for loss calculations
    segmented_tensor = torch.tensor(segmented_flat, dtype=torch.float32)
    original_tensor = torch.tensor(original_flat, dtype=torch.float32)

    # Calculate Binary Cross-Entropy Loss (BCE Loss)
    bce_loss = F.binary_cross_entropy_with_logits(segmented_tensor, original_tensor).item()

    # Calculate Dice Loss
    smooth = 1e-5
    intersection = (segmented_tensor * original_tensor).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (segmented_tensor.sum() + original_tensor.sum() + smooth)

    return accuracy, mse, bce_loss, dice_loss.item()

def process_and_evaluate_all_images(images_dir, masks_dir, model_path):
    accuracies = []
    mses = []
    bce_losses = []
    dice_losses = []

    for filename in tqdm(os.listdir(images_dir)):
        if filename.endswith('.png'):  # Adjust if images are not in PNG format
            image_path = os.path.join(images_dir, filename)
            mask_path = os.path.join(masks_dir, filename)

            try:
                # Segment the image
                segmented_image = segment_image(image_path, model_path)
                result_image = retain_dark_gray_pixels(segmented_image)

                # Load and resize the original mask
                original_mask = io.imread(mask_path)
                if original_mask.shape[2] == 4:
                    original_mask = original_mask[:, :, :3]
                if len(original_mask.shape) == 3:
                    original_mask = color.rgb2gray(original_mask) * 255
                original_mask = cv2.resize(original_mask.astype(np.uint8), (768, 1024))

                # Calculate metrics
                accuracy, mse, bce_loss, dice_loss = calculate_metrics(result_image, original_mask)
                accuracies.append(accuracy)
                mses.append(mse)
                bce_losses.append(bce_loss)
                dice_losses.append(dice_loss)
                
                print(f'Processed {filename}: Accuracy = {accuracy}, MSE = {mse}, BCE Loss = {bce_loss}, Dice Loss = {dice_loss}')
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Calculate and print overall metrics
    overall_accuracy = np.mean(accuracies)
    overall_mse = np.mean(mses)
    overall_bce_loss = np.mean(bce_losses)
    overall_dice_loss = np.mean(dice_losses)
    
    print(f'Overall Accuracy: {overall_accuracy}')
    print(f'Overall Mean Squared Error: {overall_mse}')
    print(f'Overall Binary Cross-Entropy Loss: {overall_bce_loss}')
    print(f'Overall Dice Loss: {overall_dice_loss}')

    return overall_accuracy, overall_mse, overall_bce_loss, overall_dice_loss

# Run the evaluation
overall_accuracy, overall_mse, overall_bce_loss, overall_dice_loss = process_and_evaluate_all_images(images_dir, masks_dir, model_path)
