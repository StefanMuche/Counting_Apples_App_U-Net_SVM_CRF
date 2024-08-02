import onnx
from onnx_tf.backend import prepare
onnx_model_path = 'model.onnx'
onnx_model = onnx.load(onnx_model_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model_tf.pb")
