import numpy as np
from skimage import io, img_as_float
from joblib import load
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, filters, img_as_float
from sklearn.preprocessing import scale
from joblib import dump
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
from skimage import io, color, filters, img_as_float
from skimage.color import rgb2lab
from sklearn.preprocessing import scale
import skfuzzy as fuzz
from sklearn.svm import SVC
from skimage import graph
from skimage.segmentation import relabel_sequential
import os
from tqdm import tqdm
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
image_path = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images_svm/20150921_131833_image286.png'
model_path = 'D:/Python_VSCode/licenta_v2/svm_model_red_threshold4.joblib'

def verify_red(image):
    # Normalizează imaginea
    image = img_as_float(image)
    
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    
    # Condițiile pentru a fi considerat roșu
    is_red = (red_channel > green_channel) & (red_channel > blue_channel) & (red_channel > 50 / 255.0) & (green_channel > blue_channel)
    # Atribuie valoarea 1 dacă îndeplinește condițiile, altfel 0
    red_features = np.where(is_red, 1, 0).flatten()
    
    # Adăugăm o dimensiune suplimentară pentru a obține forma (921600, 1)
    red_features = red_features[:, np.newaxis]
    
    return red_features

def segment_image(image_path, model_path):
    image = img_as_float(io.imread(image_path))
    features = verify_red(image)
    print("Features shape (before reshape):", features.shape)
    
    
    model = load(model_path)
    predicted_labels = model.predict(features)
    print("Predicted labels shape:", predicted_labels.shape)
    predicted_labels = predicted_labels.reshape(image.shape[:2])
    
        # Display the segmented image
    plt.figure(figsize=(10, 8))
    plt.imshow(predicted_labels, cmap='gray_r')
    plt.title('Segmented Image')
    plt.axis('off')
    plt.show()

segment_image(image_path, model_path)
