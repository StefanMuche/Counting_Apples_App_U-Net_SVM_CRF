import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import glob
import os
from joblib import dump, load
from tqdm import tqdm

# Calea către directoare
images_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/train_svm_50'
masks_dir = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/train_svm_masks_50'

# Obținerea listei de imagini și măști
image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
mask_paths = glob.glob(os.path.join(masks_dir, '*.jpg'))

# Funcție pentru încărcarea și preprocesarea unei imagini și a măștii sale
def load_and_preprocess_image(image_path, mask_path, image_size=(768, 1024)):
    image = io.imread(image_path)
    mask = io.imread(mask_path)
    
    # Redimensionare
    image = transform.resize(image, image_size, anti_aliasing=True)
    mask = transform.resize(mask, image_size, anti_aliasing=False, order=0)
    
    # Conversia în HSV
    hsv_image = color.rgb2hsv(image)
    
    # Extragerea caracteristicilor: luăm doar canalul de Saturare și Valoare pentru simplitate
    features = hsv_image[:, :, 1:].reshape(-1, 2)
    labels = mask.reshape(-1)
    
    return features, labels

# Pregătirea datelor
X = []
y = []

print("READ IMAGES AND MASKS...")
for img_path, mask_path in tqdm(zip(sorted(image_paths), sorted(mask_paths))):
    features, labels = load_and_preprocess_image(img_path, mask_path)
    X.append(features)
    y.append(labels)

print("DONE READING IMAGES AND MASKS...")

X = np.vstack(X)
y = np.concatenate(y)

# Împărțirea datelor în antrenament și test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("TRAIN IMAGES AND MASKS...")

# Crearea și antrenarea modelului SVM
model = svm.SVC(kernel='rbf', C=1.0)
model.fit(X_train, y_train)
print("DONE TRAINING IMAGES AND MASKS...")
# Evaluarea modelului
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:", classification_report(y_test, y_pred))

# Putem salva modelul antrenat pentru a-l folosi mai târziu
dump(model, 'svm_apple_segmenter.joblib')
