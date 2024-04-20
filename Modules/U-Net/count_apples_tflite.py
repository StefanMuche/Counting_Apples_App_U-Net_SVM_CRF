import numpy as np
import tensorflow as tf
from PIL import Image as PILImage
from skimage import measure, color
import cv2

def process_and_reconstruct_image(image_path, interpreter):
    # Încarcă imaginea folosind PIL și redimensionează la dimensiunea dorită
    image = PILImage.open(image_path).convert('RGB')
    image = image.resize((768, 1024), PILImage.BILINEAR)
    image_np = np.array(image).astype('float32') / 255.0  # Normalizare

    # Prepare the model's input dimensions (e.g., [1, 256, 256, 3] for each patch)
    input_details = interpreter.get_input_details()
    batch_size, height, width, channels = input_details[0]['shape']

    # Preallocate the reconstructed mask array
    reconstructed_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

    # Process each patch
    for i in range(0, image_np.shape[0], height):  # Assume height = 256
        for j in range(0, image_np.shape[1], width):  # Assume width = 256
            # Extract the patch
            patch = image_np[i:i+height, j:j+width]
            if patch.shape[0] != height or patch.shape[1] != width:
                continue  # Skip incomplete patches at the edges

            # Reshape patch to fit model input
            patch = np.reshape(patch, (batch_size, height, width, channels))

            # Set tensor to the correct input index and run the model
            interpreter.set_tensor(input_details[0]['index'], patch)
            interpreter.invoke()

            # Get the output and post-process it
            output_details = interpreter.get_output_details()
            pred_mask = interpreter.get_tensor(output_details[0]['index'])
            pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Thresholding to create a binary mask

            # Place the processed patch back into the reconstructed mask
            reconstructed_mask[i:i+height, j:j+width] = pred_mask[0, :, :, 0] * 255

    # Post-processing to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(reconstructed_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    labels = measure.label(opening, background=0)
    num_mere = np.max(labels)  # Count the detected objects

    # Create an overlay with labels on the original resized image
    label_overlay = color.label2rgb(labels, image=np.array(image.resize((768, 1024))), bg_label=0, alpha=0.3)

    print(f'Found {num_mere} apples.')
    return label_overlay, num_mere
