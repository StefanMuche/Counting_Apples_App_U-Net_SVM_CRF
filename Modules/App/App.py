from kivy.config import Config
Config.set('graphics', 'width', '360')  # Lățimea tipică a unui telefon mobil
Config.set('graphics', 'height', '640')  # Înălțimea tipică a unui telefon mobil

from kivy.lang import Builder
import os, sys
from kivy.resources import resource_add_path, resource_find
from kivymd.app import MDApp
from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.toolbar import MDTopAppBar, MDBottomAppBar
from kivymd.uix.filemanager import MDFileManager
from kivy.core.window import Window
from PIL import Image as PILImage
import tensorflow as tf
from kivymd.uix.label import MDLabel
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
import os, sys
from kivy.resources import resource_add_path, resource_find
import tensorflow as tf
import numpy as np
import cv2
from skimage import color, measure
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, filters, img_as_float, measure
from sklearn.preprocessing import scale
from joblib import load
import cv2
from pydensecrf import densecrf
from pydensecrf.utils import unary_from_labels, create_pairwise_gaussian, create_pairwise_bilateral

# Calea la modelul TensorFlow Lite
model_path_svm = 'D:/Python_VSCode/licenta_v2/Modules/svm_model_rbf_cielab_a_4clusters.joblib'
model_path = 'D:/Python_VSCode/licenta_v2/Models/model.tflite'
# Crearea unui interpreter TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Obține detalii despre input și output pentru a putea manipula tensorii corespunzător
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
    print("GATA")
    return output_image

def count_and_label_apples(binary_image, original_image):
    # Apply thresholding
    ret, thresh = cv2.threshold(binary_image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    
    # Apply morphological opening
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    
    # Label the connected components
    labels = measure.label(opening, background=0)
    
    # Count all unique labels
    unique_labels = np.unique(labels)
    num_apples = len(unique_labels) - 1  # Exclude the background label
    
    # Display labels on the original image
    label_overlay = color.label2rgb(labels, image=original_image, bg_label=0, alpha=0.3)
    
    return num_apples, label_overlay

def apply_crf_to_binary_image(image_path, binary_mask):
    image = img_as_float(io.imread(image_path))
    # Convert image to grayscale if it is not already
    if len(image.shape) > 2:
        image = color.rgb2gray(image)
    
    # Convert image and mask to the correct data types
    img = (image * 255).astype(np.uint8)
    binary_mask = (binary_mask / 255).astype(np.uint8)

    # Define the number of labels (2: background and foreground)
    num_labels = 2
    
    # Generate unary potentials
    labels = binary_mask
    unary = unary_from_labels(labels, num_labels, gt_prob=0.7, zero_unsure=False)

    # Create DenseCRF object
    d = densecrf.DenseCRF2D(img.shape[1], img.shape[0], num_labels)
    d.setUnaryEnergy(unary)

    # Add pairwise Gaussian potentials
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape)
    d.addPairwiseEnergy(feats, compat=3)

    # Add pairwise bilateral potentials
    img_3ch = np.stack([img]*3, axis=-1)
    feats = create_pairwise_bilateral(sdims=(4, 4), schan=(10,), img=img_3ch, chdim=2)
    d.addPairwiseEnergy(feats, compat=10)

    # Perform inference
    Q = d.inference(20)  # Increase number of iterations for better refinement
    map_result = np.argmax(Q, axis=0).reshape(img.shape)

    return map_result

def count_apples_svm(image_path, model_path):

    # Segment the image
    segmented_image = segment_image(image_path, model_path)
    result_image = retain_dark_gray_pixels(segmented_image)
    crf_refined_mask = apply_crf_to_binary_image(image_path, result_image)
    # Count apples and get labeled overlay image
    original_image = io.imread(image_path)
    num_apples, labeled_image = count_and_label_apples(crf_refined_mask, original_image)
    
    return num_apples


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

    # Post-procesare pentru a separa obiectele
    ret, thresh = cv2.threshold(reconstructed_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    labels = measure.label(opening, background=0)
    num_mere = np.max(labels)  # Numără obiectele detectate
    label_overlay = color.label2rgb(labels, image=np.array(PILImage.open(image_path).convert('RGB').resize((768, 1024))), bg_label=0, alpha=0.3)
    print(f'Found {num_mere} apples.')
    return num_mere

KV = '''
ScreenManager:
    MainScreen:
    SelectionScreen:
    UNETScreen:
    SVMScreen:
    ImageDisplayScreen:
    CameraClickScreen:

<MainScreen>:
    name: 'main'
    BoxLayout:
        orientation: 'vertical'
        MDTopAppBar:
            title: "Numararea fructelor"
            title_text_size: '10dp'
            md_bg_color: .8, 0, 0, 1  # Un roșu mai închis
            specific_text_color: 1, 1, 1, 1  # Alb pentru text
            elevation: 7
        # MDBottomAppBar:
        #     MDTopAppBar:
        #         title: "Title"
        #         icon: "git"
        #         type: "bottom"
        #         left_action_items: [["menu", lambda x: x]]
        FloatLayout:
            Image:
                source: 'bd193a193fa400756ae0fdfa0efda2c7.jpg'
                allow_stretch: True
                keep_ratio: False
                size_hint: 1, 1
                opacity: .9  
            MDRaisedButton:
                text: "Start"
                pos_hint: {"center_x": .5, "center_y": .5}
                size_hint: None, None
                size: "120dp", "48dp"
                md_bg_color: 0, 0.39, 0, 1
                on_release: app.root.current = 'selection'

<SelectionScreen>:
    name: 'selection'
    BoxLayout:
        orientation: 'vertical'
        MDTopAppBar:
            id: top_bar
            title: "Alegerea algoritmului"
            md_bg_color: .8, 0, 0, 1  # Un roșu mai închis
            specific_text_color: 1, 1, 1, 1  # Alb pentru text
            elevation: 7
        FloatLayout:
            Image:
                source: 'bd193a193fa400756ae0fdfa0efda2c7.jpg'
                allow_stretch: True
                keep_ratio: False
                size_hint: 1, 1
                opacity: .9  
            MDRaisedButton:
                text: "U-Net"
                pos_hint: {"center_x": .5, "center_y": .6}
                size_hint: None, None
                size: "120dp", "48dp"
                md_bg_color: 0, 0.39, 0, 1  # Verde închis
                on_release: app.set_algorithm('unet')
            MDRaisedButton:
                text: "SVM"
                pos_hint: {"center_x": .5, "center_y": .4}
                size_hint: None, None
                size: "120dp", "48dp"
                md_bg_color: 0, 0.39, 0, 1  # Verde închis
                on_release: app.set_algorithm('svm')
            MDRaisedButton:
                text: 'Back'
                pos_hint: {"center_x": .5, "center_y": .1}
                size_hint: None, None
                size: "120dp", "48dp"
                md_bg_color: 0, 0.39, 0, 1  # Verde închis
                on_release: app.root.current = 'main'


<UNETScreen>:
    name: 'ChoosePhoto-U-NET'
    BoxLayout:
        orientation: 'vertical'
        MDTopAppBar:
            title: "U-NET"
            md_bg_color: .8, 0, 0, 1  # Un roșu mai închis
            specific_text_color: 1, 1, 1, 1  # Alb pentru text
            elevation: 7 
        FloatLayout:
            Image:
                source: 'bd193a193fa400756ae0fdfa0efda2c7.jpg'
                allow_stretch: True
                keep_ratio: False
                size_hint: 1, 1
                opacity: .9 
            MDRaisedButton:
                text: "Choose a photo"
                pos_hint: {"center_x": .5, "center_y": .5}
                size_hint: None, None
                size: "160dp", "60dp"
                md_bg_color: 0, 0.39, 0, 1
                on_release: app.open_file_manager()
            MDRaisedButton:
                text: 'Back'
                pos_hint: {"center_x": .5, "center_y": .1}
                size_hint: None, None
                size: "120dp", "48dp"
                md_bg_color: 0, 0.39, 0, 1
                on_release: app.root.current = 'selection'

<SVMScreen>:
    name: 'ChoosePhoto-SVM'
    BoxLayout:
        orientation: 'vertical'
        MDTopAppBar:
            title: "SVM"
            md_bg_color: .8, 0, 0, 1  # Un roșu mai închis
            specific_text_color: 1, 1, 1, 1  # Alb pentru text
            elevation: 7 
        FloatLayout:
            Image:
                source: 'bd193a193fa400756ae0fdfa0efda2c7.jpg'
                allow_stretch: True
                keep_ratio: False
                size_hint: 1, 1
                opacity: .9  
            MDRaisedButton:
                text: "Choose a photo"
                pos_hint: {"center_x": .5, "center_y": .5}
                size_hint: None, None
                size: "160dp", "60dp"
                md_bg_color: 0, 0.39, 0, 1
                on_release: app.open_file_manager()
            MDRaisedButton:
                text: 'Back'
                pos_hint: {"center_x": .5, "center_y": .1}
                size_hint: None, None
                size: "120dp", "48dp"
                md_bg_color: 0, 0.39, 0, 1
                on_release: app.root.current = 'selection'

<ImageDisplayScreen>:
    name: 'imagedisplay'
    BoxLayout:
        orientation: 'vertical'
        MDTopAppBar:
            title: "Vizualizare Imagine"
            md_bg_color: .8, 0, 0, 1
            specific_text_color: 1, 1, 1, 1
            elevation: 7
        ScrollView:
            GridLayout:
                id: image_grid
                cols: 1
                size_hint_y: None
                height: self.minimum_height
                spacing: dp(30)
        MDRaisedButton:
            text: "Înapoi"
            pos_hint: {"center_x": .5, "center_y": .5}
            size_hint: None, None
            size: "140dp", "52dp"  
            md_bg_color: 0, 0.39, 0, 1
            # Efect 3D
            elevation_normal: 2
            elevation_down: 5
            # Colturi rotunjite
            radius: [45]
            on_release: app.root.current = 'main'

'''

class MainScreen(MDScreen):
    pass

class SelectionScreen(MDScreen):
    pass

class UNETScreen(MDScreen):
    pass

class SVMScreen(MDScreen):
    pass

class ImageDisplayScreen(MDScreen):
    pass

class CameraClickScreen(MDScreen):
    pass

class MyApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)
        self.manager = None
        self.selected_files = []
        self.total_mere = 0
        self.current_algorithm = None
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.add_path_to_selection,
            preview=True,
        )
    
    def build(self):
        return Builder.load_string(KV)

    def on_start(self):
        # Set extensions for images
        self.file_manager.ext = ['.jpg', '.jpeg', '.png']  

    def set_algorithm(self, algorithm):
        self.current_algorithm = algorithm
        if algorithm == 'unet':
            self.root.current = 'ChoosePhoto-U-NET'
        elif algorithm == 'svm':
            self.root.current = 'ChoosePhoto-SVM'

    def open_file_manager(self, *args):
        # You can specify the start directory here
        self.file_manager.show('/')  
        self.manager_open = True
    
    def add_path_to_selection(self, path):
        valid_extensions = ('.jpg', '.jpeg', '.png')

        if path.endswith(valid_extensions) and path not in self.selected_files:
            self.selected_files.append(path)
        else:
            self.process_images()
            self.manager_open = False
            self.file_manager.close()

    def process_images(self):
        results = []  # List to hold (path, count) tuples
        for path in self.selected_files:
            if self.current_algorithm == 'unet':
                num_mere = process_and_reconstruct_image(path, interpreter)
                img = path
            elif self.current_algorithm == 'svm':
                num_mere = count_apples_svm(path, model_path_svm)  # Use the count_apples function for SVM
                img = path  # The image path itself, as count_apples_svm doesn't return the image
            results.append((path, num_mere))

        # Clear and update UI for each image and count
        screen = self.root.get_screen('imagedisplay')
        screen.ids.image_grid.clear_widgets()
        for image_path, count in results:
            image_box = BoxLayout(orientation='vertical', size_hint_y=None, height=200 + 80)  # Adjust as necessary
            img_widget = Image(source=image_path, size_hint=(1, None), height=280)  # Image takes up all width, height is fixed
            count_label = MDLabel(text=f"{count} mere", size_hint_y=None, height=20, halign='center')
            # Bind label size to texture size and center text horizontally
            count_label.bind(size=count_label.setter('text_size'))  
            count_label.bind(texture_size=count_label.setter('size'))
            
            image_box.add_widget(img_widget)
            image_box.add_widget(count_label)
            screen.ids.image_grid.add_widget(image_box)

        # Add total count at the end
        total_mere = sum([count for _, count in results])
        total_count_label = MDLabel(text=f"În total sunt {total_mere} mere.", size_hint_y=None, height=20, halign='center')
        # Same binding for the total count label
        total_count_label.bind(size=total_count_label.setter('text_size'))
        total_count_label.bind(texture_size=total_count_label.setter('size'))
        screen.ids.image_grid.add_widget(total_count_label)
        self.root.current = 'imagedisplay'
        self.reset_selection()
        
    def reset_selection(self):
        # Clear the selected files list for new selections
        self.selected_files = []

    def exit_manager(self, *args):
        # Close the file manager
        self.manager_open = False
        self.file_manager.close()

    def events(self, instance, keyboard, keycode, text, modifiers):
        # Close file manager on back button (Android)
        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()


if __name__ == '__main__':
    try:
        if hasattr(sys, '_MEIPASS'):
            resource_add_path(os.path.join(sys._MEIPASS))
        app = MyApp()
        app.run()
    except Exception as e:
        print(e)
        input("Press enter.")