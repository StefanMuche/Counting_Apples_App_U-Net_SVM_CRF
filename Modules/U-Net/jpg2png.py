import os
import cv2

def convert_images_to_png_with_replacement(input_dir, output_dir, replacement_from, replacement_to):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.png') or filename.endswith('.jpeg'):
            # Construct the full file path
            input_path = os.path.join(input_dir, filename)
            
            # Load the image
            image = cv2.imread(input_path)
            
            if image is None:
                print(f"Failed to load image: {input_path}")
                continue

            # Replace the specified substring in the filename
            new_filename = filename.replace(replacement_from, replacement_to)
            base_filename = os.path.splitext(new_filename)[0]
            output_path = os.path.join(output_dir, base_filename + '.png')
            
            # Save the image in PNG format
            cv2.imwrite(output_path, image)
            
            print(f"Converted {filename} to PNG and saved as {base_filename}.png")

# Example usage
input_directory = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/Masks_for_erorrs_png'
output_directory = 'D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/train/Masks_for_errors_png_2'

replacement_from = 'mask'
replacement_to = 'image'

convert_images_to_png_with_replacement(input_directory, output_directory, replacement_from, replacement_to)
