import tensorflow as tf
import numpy as np
import cv2
from skimage import color, measure
import matplotlib.pyplot as plt
from skimage import io, filters, img_as_float
from sklearn.metrics import confusion_matrix
from joblib import load
from pydensecrf import densecrf
from pydensecrf.utils import unary_from_labels, create_pairwise_gaussian, create_pairwise_bilateral
from PIL import Image as PILImage
import os
from tqdm import tqdm

# Directories
images_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images_for_errors_png'
masks_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/Masks_for_errors'

# Load the TFLite model
model_path = 'D:/Python_VSCode/licenta_v2/Models/model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def process_and_reconstruct_image(image_path, interpreter):
    # Încarcă imaginea originală și convert-o în formatul așteptat de TensorFlow
    image = PILImage.open(image_path).convert('RGB')
    image = image.resize((768, 1024), PILImage.BILINEAR)
    image_np = np.array(image).astype('float32') / 255.0  # Normalizare și convertire în NumPy array

    # Transpunere pentru a aduce canalele în conformitate cu așteptările modelului
    image_np = np.transpose(image_np, (2, 0, 1))  # De la (H, W, C) la (C, H, W)
    # Prepare the model's input dimensions (e.g., [1, 256, 256, 3] for each patch)
    input_details = interpreter.get_input_details()
    batch_size, height, width, channels = input_details[0]['shape']

    # Inițializează tensorul pentru imaginea reconstituită
    reconstructed_mask = np.zeros((1024, 768), dtype=np.uint8)

    # Imparte imaginea în patch-uri și procesează fiecare patch
    for i in range(4):  # 4 patch-uri pe înălțime
        for j in range(3):  # 3 patch-uri pe lățime
            # Extrage patch-ul
            patch = image_np[:, i*256:(i+1)*256, j*256:(j+1)*256]
            patch = np.expand_dims(patch, axis=0)  # Adaugă dimensiunea batch-ului

            # # Aplică modelul TensorFlow pe patch, folosind dicționarul de intrare
            # input_dict = {'input': patch}  # 'input' este numele tensorului de intrare specificat în model
            # results = model(**input_dict)
            # pred_mask = results['output'].numpy()  # 'output' este numele tensorului de ieșire
            # pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Convertirea predicției în masca binară
             # Set tensor to the correct input index and run the model
            interpreter.set_tensor(input_details[0]['index'], patch)
            interpreter.invoke()

            # Get the output and post-process it
            output_details = interpreter.get_output_details()
            pred_mask = interpreter.get_tensor(output_details[0]['index'])
            pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Thresholding to create a binary mask

            # Adaugă patch-ul procesat în masca reconstituită
            reconstructed_mask[i*256:(i+1)*256, j*256:(j+1)*256] = pred_mask[0, 0, :, :] * 255

    return reconstructed_mask

confusion_mats = []

for image_name in tqdm(os.listdir(images_dir)):
    image_path = os.path.join(images_dir, image_name)
    mask_path = os.path.join(masks_dir, image_name)

    if not os.path.exists(mask_path):
        continue

    pred_mask = process_and_reconstruct_image(image_path, interpreter)
    true_mask = np.array(PILImage.open(mask_path).convert('L')).astype(np.uint8)

    # Calculate confusion matrix
    conf_mat = confusion_matrix(true_mask.flatten(), pred_mask.flatten(), labels=[0, 255])
    conf_mat = conf_mat / true_mask.size  # Normalize
    confusion_mats.append(conf_mat)

# Calculate mean confusion matrix
total_confusion = np.mean(np.array(confusion_mats), axis=0)

print("Average Confusion Matrix:")
print(total_confusion)
