from kivy.config import Config
Config.set('graphics', 'width', '360')  # Lățimea tipică a unui telefon mobil
Config.set('graphics', 'height', '640')  # Înălțimea tipică a unui telefon mobil

from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.toolbar import MDTopAppBar, MDBottomAppBar

KV = '''
ScreenManager:
    MainScreen:
    SelectionScreen:

<MainScreen>:
    name: 'main'
    BoxLayout:
        orientation: 'vertical'
        MDTopAppBar:
            title: "Segmentarea și numărarea automata automata a fructelor"
            title_text_size: '10dp'
            md_bg_color: 1, 0, 0, 1
            specific_text_color: 1, 1, 1, 1
            elevation: 7
        # MDBottomAppBar:
        #     MDTopAppBar:
        #         title: "Title"
        #         icon: "git"
        #         type: "bottom"
        #         left_action_items: [["menu", lambda x: x]]
        FloatLayout:
            MDRaisedButton:
                text: "Start"
                pos_hint: {"center_x": .5, "center_y": .5}
                size_hint: None, None
                size: "120dp", "48dp"
                md_bg_color: 1, 0, 0, 1
                on_release: app.root.current = 'selection'

<SelectionScreen>:
    name: 'selection'
    BoxLayout:
        orientation: 'vertical'
        MDTopAppBar:
            title: "Alegerea algoritmului"
            md_bg_color: 1, 0, 0, 1
            specific_text_color: 1, 1, 1, 1
            elevation: 7
        # MDBottomAppBar:
        #     MDTopAppBar:
        #         title: "Title"
        #         icon: "git"
        #         type: "bottom"
        #         left_action_items: [["menu", lambda x: x]]
        FloatLayout:
            MDRaisedButton:
                text: "U-Net"
                pos_hint: {"center_x": .5, "center_y": .6}
                size_hint: None, None
                size: "120dp", "48dp"
                md_bg_color: 1, 0, 0, 1
            MDRaisedButton:
                text: "SVM"
                pos_hint: {"center_x": .5, "center_y": .4}
                size_hint: None, None
                size: "120dp", "48dp"
                md_bg_color: 1, 0, 0, 1
            MDRaisedButton:
                text: 'Back'
                pos_hint: {"center_x": .5, "center_y": .1}
                size_hint: None, None
                size: "120dp", "48dp"
                md_bg_color: 1, 0, 0, 1
                on_release: app.root.current = 'main'
'''

class MainScreen(MDScreen):
    pass

class SelectionScreen(MDScreen):
    pass

class MyApp(MDApp):
    def build(self):
        # self.theme_cls.primary_palette = "Blue"  # Alegeți o schemă de culori
        # self.theme_cls.theme_style = "Light"  # Alegeți între "Light" și "Dark"
        return Builder.load_string(KV)

MyApp().run()
