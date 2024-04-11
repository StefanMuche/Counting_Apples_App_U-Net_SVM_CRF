import onnx
from onnx_tf.backend import prepare
onnx_model_path = 'model.onnx'
# Încarcă modelul ONNX
onnx_model = onnx.load(onnx_model_path)

# Convertirea modelului ONNX în TensorFlow
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model_tf.pb")
