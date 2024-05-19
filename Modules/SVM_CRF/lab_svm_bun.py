import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, filters, img_as_float
from sklearn.preprocessing import scale
from joblib import load

image_path = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images_svm/20150921_131625_image471.png'
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

# Segment the image
segmented_image = segment_image(image_path, model_path)
result_image = retain_dark_gray_pixels(segmented_image)

plt.figure(figsize=(10, 8))
plt.imshow(segmented_image, cmap='gray')
plt.title('Retained Dark Gray Pixels')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 8))
plt.imshow(result_image, cmap='gray')
plt.title('Retained Dark Gray Pixels')
plt.axis('off')
plt.show()
