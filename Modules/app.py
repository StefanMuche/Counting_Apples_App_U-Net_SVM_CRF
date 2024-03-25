from kivy.app import App
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput

Window.clearcolor = (1, 0, 0, 1)
Window.size = (360, 600)

# class MainApp(App):
#     def build(self):
#         button = Button(text = 'Print this', size_hint = (0.8, 0.2), font_size = '20sp', 
#                         pos_hint = {'center_x':0.6, 'center_y':0.8}, on_press = self.printpress, on_release = self.printrelease)
#         return button
    
#     def printpress(self, obj):
#         print("Button has been pressed")

#     def printrelease(self, obj):
#         print('Button has been released')


# class MainApp(App):
#     def build(self):
#         layout = BoxLayout(orientation = 'vertical', spacing = 10, padding = 40)
#         button = Button(text = 'Button1')
#         button2 = Button(text = 'Button2')
#         button3 = Button(text = 'Button3')
#         layout.add_widget(button)
#         layout.add_widget(button2)
#         layout.add_widget(button3)
#         return layout
    

# class MainApp(App):
#     def build(self):
#         layout = BoxLayout(orientation = 'vertical', spacing = 100, padding = 40)
#         img = Image(source = 'pngtree-korean-bear-sticker-cute-peek-drawing-png-image_3989068.png')
#         button = Button(text = 'Login', size_hint = (None, None), width = 100, height = 50, pos_hint = {'center_x': 0.5, 'center_y': 0})
#         layout.add_widget(img)
#         layout.add_widget(button)
#         return layout
    
# class MainApp(App):
#     def build(self):
#         layout = GridLayout(cols = 2)
#         btn1 = Button(text = 'Hello 1')
#         btn2 = Button(text = 'Hello 2')

#         btn3 = Button(text = 'Hello 3')
#         btn4 = Button(text = 'Hello 4')
#         layout.add_widget(btn1)
#         layout.add_widget(btn2)
#         layout.add_widget(btn3)
#         layout.add_widget(btn4)
#         return layout

class MainApp(App):
    def build(self):
        layout = GridLayout(cols = 2, row_force_default= True, row_default_height = 40, spacing = 20, padding = 30)
        self.weight = TextInput(text = 'Enter the weight here')
        self.height = TextInput(text = 'Enter the height here')
        submit = Button(text = 'Submit', on_press = self.submit)
        layout.add_widget(self.weight)
        layout.add_widget(self.height)
        layout.add_widget(submit)
        return layout

    def submit(self, obj):
        print(f"Weight: {self.weight.text}")
        print(f"Height: {self.height.text}")

MainApp().run()