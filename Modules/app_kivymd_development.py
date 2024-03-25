from kivymd.app import MDApp
from kivy.lang import Builder

screen_helper = '''
Screen:
    BoxLayout:
        orientation: 'vertical'
        MDToolBar: 
            title: 'Demo Application'
        MDLabel:
            text: 'Hello World'
            hialign: 'center'
'''


class DemoApp(MDApp):
    def build(self):
        screen = Builder.load_string(screen_helper)
        return screen
    
DemoApp().run()