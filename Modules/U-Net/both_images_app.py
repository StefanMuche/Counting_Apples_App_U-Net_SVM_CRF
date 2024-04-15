from kivy.config import Config
Config.set('graphics', 'width', '360')  # Lățimea tipică a unui telefon mobil
Config.set('graphics', 'height', '640')  # Înălțimea tipică a unui telefon mobil

from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.toolbar import MDTopAppBar, MDBottomAppBar
from kivymd.uix.filemanager import MDFileManager
from kivy.core.window import Window
from PIL import Image as PILImage
from count_apples_tf import process_and_reconstruct_image
import numpy as np
from kivy.metrics import dp
import tensorflow as tf
from kivymd.uix.label import MDLabel
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from plyer import camera
import time
import os
from PIL import Image as PILImage

########## INCARCAREA MODELULUI ########################################
model_path = 'D:/Python_VSCode/licenta_v2/Modules/U-Net/model_tf.pb'  # Ajustează calea către modelul salvat
model = tf.saved_model.load(model_path)

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
            title: "Segmentarea și numărarea automata automata a fructelor"
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
                opacity: .9  # ajustează pentru a asigura că textul este lizibil
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
                opacity: .9  # ajustează pentru a asigura că textul este lizibil
            MDRaisedButton:
                text: "U-Net"
                pos_hint: {"center_x": .5, "center_y": .6}
                size_hint: None, None
                size: "120dp", "48dp"
                md_bg_color: 0, 0.39, 0, 1  # Verde închis
                on_release: app.root.current = 'ChoosePhoto-U-NET'
            MDRaisedButton:
                text: "SVM"
                pos_hint: {"center_x": .5, "center_y": .4}
                size_hint: None, None
                size: "120dp", "48dp"
                md_bg_color: 0, 0.39, 0, 1  # Verde închis
                on_release: app.root.current = 'ChoosePhoto-SVM'
            MDRaisedButton:
                text: 'Back'
                pos_hint: {"center_x": .5, "center_y": .1}
                size_hint: None, None
                size: "120dp", "48dp"
                md_bg_color: 0, 0.39, 0, 1  # Verde închis
                on_release: app.root.current = 'main'


<UNETScreen>
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
                opacity: .9  # ajustează pentru a asigura că textul este lizibil
            MDRaisedButton:
                text: "Choose a photo"
                pos_hint: {"center_x": .5, "center_y": .6}
                size_hint: None, None
                size: "120dp", "48dp"
                md_bg_color: 0, 0.39, 0, 1
                on_release: app.open_file_manager()
            MDRaisedButton:
                text: "Take a photo"
                pos_hint: {"center_x": .5, "center_y": .4}
                size_hint: None, None
                size: "120dp", "48dp"
                md_bg_color: 0, 0.39, 0, 1
            MDRaisedButton:
                text: 'Back'
                pos_hint: {"center_x": .5, "center_y": .1}
                size_hint: None, None
                size: "120dp", "48dp"
                md_bg_color: 0, 0.39, 0, 1
                on_release: app.root.current = 'selection'

<SVMScreen>
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
                opacity: .9  # ajustează pentru a asigura că textul este lizibil
            MDRaisedButton:
                text: "Choose a photo"
                pos_hint: {"center_x": .5, "center_y": .6}
                size_hint: None, None
                size: "120dp", "48dp"
                md_bg_color: 0, 0.39, 0, 1
            MDRaisedButton:
                text: "Take a photo"
                pos_hint: {"center_x": .5, "center_y": .4}
                size_hint: None, None
                size: "120dp", "48dp"
                md_bg_color: 0, 0.39, 0, 1
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
                spacing: dp(30)  # Reduced spacing
        BoxLayout:
            size_hint_y: None
            height: '52dp'
            pos_hint: {'center_x': 0.5}
            MDRaisedButton:
                text: "Înapoi"
                pos_hint: {"center_x": .5, "center_y": .5}
                on_release: app.root.current = 'main'
                size_hint: None, None
                size: "140dp", "52dp"
                md_bg_color: 0, 0.39, 0, 1
                elevation_normal: 2
                elevation_down: 5
                radius: [45]

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
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.add_path_to_selection,
            preview=True,
        )
    
    def build(self):
        return Builder.load_string(KV)

    def on_start(self):
        self.file_manager.ext = ['.jpg', '.jpeg', '.png']  # Set extensions for images

    def open_file_manager(self, *args):
        self.file_manager.show('/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/test/images')  # You can specify the start directory here
        self.manager_open = True
    
    def add_path_to_selection(self, path):
        if path not in self.selected_files:
            self.selected_files.append(path)  # Add unique file to the list

    def process_images(self):
        results = []  # List to hold (original_path, segmented_path, count) tuples
        # ... [rest of your code] ...
        for path in self.selected_files:
            img, num_mere = process_and_reconstruct_image(path, model)
            if img.dtype != np.uint8:
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

            pil_image = PILImage.fromarray(img)
            segmented_path = os.path.splitext(path)[0] + '_segmented.png'
            pil_image.save(segmented_path)
            results.append((path, segmented_path, num_mere))
        screen = self.root.get_screen('imagedisplay')
        screen.ids.image_grid.clear_widgets()
        for original_path, segmented_path, count in results:
            # Define a fixed size for images and labels
            img_size = (dp(200), dp(200))  # Update this size to fit your need
            label_size = (dp(100), dp(50))  # Update this size to fit your need
            count_label = MDLabel(text=f"{count} mere", size_hint_y=None, height=20, halign='center')
            # Bind label size to texture size and center text horizontally
            count_label.bind(size=count_label.setter('text_size'))  
            count_label.bind(texture_size=count_label.setter('size'))

            # Horizontal BoxLayout for each image pair
            image_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(250))

            # Original image widget
            original_img_widget = Image(
                source=original_path,
                size_hint=(1, None),
                size=img_size,
                allow_stretch=True,
                keep_ratio=False
            )

            # Segmented image widget
            segmented_img_widget = Image(
                source=segmented_path,
                size_hint=(1, None),
                size=img_size,
                allow_stretch=True,
                keep_ratio=False,
            )

            # Label for the count, centered
            # count_label = MDLabel(
            #     text=f"{count} mere",
            #     size_hint=(None, None),
            #     size=label_size,
            #     halign='center',
            #     valign='center'
            # )
            # count_label.bind(size=count_label.setter('text_size'))

            # Add widgets to the horizontal BoxLayout
            image_box.add_widget(original_img_widget)
            image_box.add_widget(segmented_img_widget)
            image_box.add_widget(count_label)
            # image_box.add_widget(count_label)

            # Add the horizontal BoxLayout to the GridLayout
            screen.ids.image_grid.add_widget(image_box)

        # ... [the rest of your existing code for adding the total count] ...


        # Add the total count at the end outside the scroll view
        total_mere = sum([count for _, _, count in results])
        total_count_label = MDLabel(
            text=f"În total sunt {total_mere} mere.",
            size_hint=(1, None),
            height=dp(20),
            halign='center'
        )
        screen.ids.image_grid.add_widget(total_count_label)
        self.root.current = 'imagedisplay'
        self.reset_selection()

        
    def reset_selection(self):
        # Clear the selected files list for new selections
        self.selected_files = []

    def exit_manager(self, *args):
        # Close the file manager
        self.process_images()
        self.manager_open = False
        self.file_manager.close()

    def events(self, instance, keyboard, keycode, text, modifiers):
        # Close file manager on back button (Android)
        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()


MyApp().run()