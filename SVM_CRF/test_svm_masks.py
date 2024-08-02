import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform
from sklearn import svm
from joblib import dump, load

# Încărcarea modelului antrenat
model = load('svm_apple_segmenter.joblib')

# Funcția pentru încărcarea și preprocesarea imaginii
def preprocess_image(image_path, image_size=(768, 1024)):
    image = io.imread(image_path)
    image = transform.resize(image, image_size, anti_aliasing=True)
    hsv_image = color.rgb2hsv(image)
    features = hsv_image[:, :, 1:].reshape(-1, 2)  # Folosim canalul de Saturare și Valoare
    return features, hsv_image.shape

# Funcția pentru segmentarea imaginii
def segment_image(image_path):
    features, original_shape = preprocess_image(image_path)
    predicted_labels = model.predict(features)
    segmented_image = predicted_labels.reshape(original_shape[:2])
    return segmented_image

# Încărcarea unei imagini noi
image_path = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images_svm/20150921_132038_image271.png'
segmented_image = segment_image(image_path)

# Afișarea rezultatului
plt.figure(figsize=(10, 5))
plt.imshow(segmented_image, cmap='gray')
plt.title('Segmented Image')
plt.show()
