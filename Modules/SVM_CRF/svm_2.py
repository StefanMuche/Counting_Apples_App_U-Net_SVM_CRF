import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure
from skimage.morphology import closing, square
from skimage.segmentation import clear_border

# Încărcarea imaginii
image_path = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images_svm/20150921_132038_image271.png'  # Schimbă acest path cu locația reală a imaginii tale
image = io.imread(image_path)
image_gray = color.rgb2gray(image)

# Binarizarea imaginii
thresh = filters.threshold_otsu(image_gray)  # Calculul unui prag folosind metoda Otsu
binary_image = closing(image_gray > thresh, square(3))  # Aplicarea închiderii pentru a elimina găurile mici
cleared_image = clear_border(binary_image)  # Opțional: eliminarea obiectelor care ating marginea imaginii

# Aplicarea funcției label
labeled_image, num_features = measure.label(cleared_image, return_num=True, connectivity=2)

# Vizualizarea imaginilor
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(binary_image, cmap='gray')
axes[1].set_title('Binary Image')
axes[1].axis('off')

axes[2].imshow(labeled_image, cmap='nipy_spectral')
axes[2].set_title('Labeled Image')
axes[2].axis('off')

plt.show()
