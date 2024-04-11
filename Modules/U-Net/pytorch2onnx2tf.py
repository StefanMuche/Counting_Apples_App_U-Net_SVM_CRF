import tensorflow as tf

# Calea către directorul modelului salvat
model_dir = 'D:/Python_VSCode/licenta_v2/Modules/U-Net/model_tf.pb'

# Încărcarea modelului
loaded_model = tf.saved_model.load(model_dir)

# Crearea unui convertor pentru modelul încărcat
converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)

# (Opțional) Setarea optimizărilor
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convertirea modelului
tflite_model = converter.convert()

# Salvarea modelului TFLite într-un fișier
tflite_model_path = 'model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)