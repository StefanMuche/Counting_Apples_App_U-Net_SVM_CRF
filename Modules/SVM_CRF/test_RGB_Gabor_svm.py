from skimage import io, color, exposure
import matplotlib.pyplot as plt
import numpy as np

def balance_brightness(image_path):
    # Load the image
    image = io.imread(image_path)
    
    # Convert the image from RGB to HSV
    hsv = color.rgb2hsv(image)
    
    # Normalize the V channel
    v = hsv[:, :, 2]
    v = exposure.rescale_intensity(v, in_range='image', out_range=(0, 1))
    
    # Merge the HSV channels back
    hsv[:, :, 2] = v
    balanced_image = color.hsv2rgb(hsv)
    
    # Display the original and balanced images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Balanced Image')
    plt.imshow(balanced_image)
    plt.axis('off')
    
    plt.show()
    
    # Save the balanced image
    output_path = 'balanced_image.png'
    io.imsave(output_path, (balanced_image * 255).astype(np.uint8))
    return output_path

# Call the function with the provided image path
image_path = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/images_svm/20150921_131453_image281.png'
balanced_image_path = balance_brightness(image_path)
# print(f'Balanced image saved at: {balanced_image_path}')
