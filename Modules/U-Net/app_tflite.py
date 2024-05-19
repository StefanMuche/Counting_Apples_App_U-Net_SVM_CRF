
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
from count_apples_tflite import process_and_reconstruct_image
import tensorflow as tf
from kivymd.uix.label import MDLabel
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout

########## INCARCAREA MODELULUI ########################################
import tensorflow as tf

# Calea la modelul TensorFlow Lite
model_path = 'D:/Python_VSCode/licenta_v2/Models/model.tflite'

# Crearea unui interpreter TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Obține detalii despre input și output pentru a putea manipula tensorii corespunzător
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


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
            img, num_mere = process_and_reconstruct_image(path, interpreter)
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


MyApp().run()
