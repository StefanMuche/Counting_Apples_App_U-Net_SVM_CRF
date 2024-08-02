import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from unet_pytorch import UNet
import matplotlib.pyplot as plt
from skimage import color, measure
import tensorflow as tf
def process_and_reconstruct_image_and_show_opening(image_path, interpreter):
    # Load and preprocess the original image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((768, 1024), Image.BILINEAR)
    image_np = np.array(image).astype('float32') / 255.0

    # Transpose the image for the model
    image_np = np.transpose(image_np, (2, 0, 1))

    # Prepare the model's input dimensions
    input_details = interpreter.get_input_details()
    batch_size, height, width, channels = input_details[0]['shape']

    # Initialize the tensor for the reconstructed image
    reconstructed_mask = np.zeros((1024, 768), dtype=np.uint8)

    # Process each patch of the image
    for i in range(4):
        for j in range(3):
            patch = image_np[:, i*256:(i+1)*256, j*256:(j+1)*256]
            patch = np.expand_dims(patch, axis=0)

            # Run the model on the patch
            interpreter.set_tensor(input_details[0]['index'], patch)
            interpreter.invoke()

            # Get the output and post-process it
            output_details = interpreter.get_output_details()
            pred_mask = interpreter.get_tensor(output_details[0]['index'])
            pred_mask = (pred_mask > 0.5).astype(np.uint8)

            # Add the processed patch to the reconstructed mask
            reconstructed_mask[i*256:(i+1)*256, j*256:(j+1)*256] = pred_mask[0, 0, :, :] * 255

    # Post-processing to separate objects
    ret, thresh = cv2.threshold(reconstructed_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    labels = measure.label(opening, background=0)
    num_mere = np.max(labels)
    label_overlay = color.label2rgb(labels, image=np.array(Image.open(image_path).convert('RGB').resize((768, 1024))), bg_label=0, alpha=0.3)

    print(f'Found {num_mere} apples.')

    # Display the opening result
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(opening, cmap='gray')
    plt.title('Opening')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(label_overlay)
    plt.title('Label Overlay')
    plt.axis('off')

    plt.show()

# Define the image path and interpreter (replace with your actual paths and interpreter setup)
image_path = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images_for_errors_png/resized_image_543.png'
model_path = 'D:/Python_VSCode/licenta_v2/Models/model.tflite'
# Crearea unui interpreter TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
# Visualize the opening
process_and_reconstruct_image_and_show_opening(image_path, interpreter)
