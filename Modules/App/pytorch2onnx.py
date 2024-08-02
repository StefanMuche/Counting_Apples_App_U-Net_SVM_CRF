import torch
import torch.onnx
from unet_pytorch import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(pretrained=True, out_channels=1)
model_path = "D:/Licenta-Segmentarea si numararea automata a fructelor/Datasets/modele/unet_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Presupunem că 'model' este deja încărcat și pregătit
dummy_input = torch.randn(1, 3, 256, 256)  # Asumăm că intrarea în model este de dimensiune 256x256 cu 3 canale
onnx_model_path = "model.onnx"

torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True, input_names=['input'], output_names=['output'])
