from kivy.config import Config
Config.set('graphics', 'width', '360')  # Lățimea tipică a unui telefon mobil
Config.set('graphics', 'height', '640')  # Înălțimea tipică a unui telefon mobil

from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.toolbar import MDTopAppBar, MDBottomAppBar
from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.filemanager import MDFileManager
from kivy.core.window import Window
from PIL import Image as PILImage
import os
from count_apples import process_and_reconstruct_image
from unet_pytorch import UNet
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from unet_pytorch import UNet
import matplotlib.pyplot as plt
from skimage import color, measure

transform = transforms.Compose([
    transforms.ToTensor(),
])

########## INCARCAREA MODELULUI ########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(pretrained=True, out_channels=1)
model_path = "D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/modele/unet_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


KV = '''
ScreenManager:
    MainScreen:
    SelectionScreen:
    UNETScreen:
    SVMScreen:
    ImageDisplayScreen:

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
            md_bg_color: .8, 0, 0, 1  # Un roșu mai închis
            specific_text_color: 1, 1, 1, 1  # Alb pentru text
            elevation: 7
        ScrollView:
            BoxLayout:
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                Image:
                    id: original_image
                    source: ''
                    size_hint: 1, None
                    height: dp(300)
                    allow_stretch: True
                    keep_ratio: True
                Image:
                    id: gray_image
                    source: ''
                    size_hint: 1, None
                    height: dp(300)
                    allow_stretch: True
                    keep_ratio: True
                MDLabel:
                    id: fruit_count
                    text: ''
                    halign: 'center'
                    size_hint_y: None
                    height: self.texture_size[1]    
        MDRaisedButton:
            text: "Înapoi"
            pos_hint: {"center_x": .5}
            size_hint: None, None
            size: "120dp", "48dp"
            md_bg_color: 0, 0.39, 0, 1
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

class MyApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)
        self.manager = None
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path,
            preview=True,
        )

    def build(self):
        return Builder.load_string(KV)

    def on_start(self):
        self.file_manager.ext = ['.jpg', '.jpeg', '.png']  # Set extensions for images

    def open_file_manager(self, *args):
        self.file_manager.show('/Licenta-Segmentarea si numararea automata a fructelor/Datasets/detection/test/images')  # You can specify the start directory here
        self.manager_open = True

    def select_path(self, path):
        # img = PILImage.open(path).convert('L')
        img, num_mere = process_and_reconstruct_image(path, model, device, transform)
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

        pil_image = Image.fromarray(img)
        # pil_image = Image.fromarray(img)
        gray_path = os.path.splitext(path)[0] + '_gray.png'
        pil_image.save(gray_path)
        
        # Actualizează sursele widget-urilor Image pentru a afișa imaginile
        screen = self.root.get_screen('imagedisplay')
        screen.ids.original_image.source = path
        # screen.ids.original_image.reload()  # Reîncarcă imaginea dacă a fost vizualizată anterior
        
        screen.ids.gray_image.source = gray_path
        # screen.ids.gray_image.reload()  # Reîncarcă imaginea
        
        # Actualizează textul MDLabel cu numărul de fructe detectate
        screen = self.root.get_screen('imagedisplay')
        screen.ids.fruit_count.text = f"În imagine sunt {num_mere} fructe."

        self.root.current = 'imagedisplay'
        self.exit_manager()

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
