import torch
import onnx
from onnx2keras import onnx_to_keras
import timm

def convert_to_keras(model_path):
    # Loads pytorch model

    model = timm.create_model('ig_resnext101_32x8d', pretrained=True)
    model.eval()

    x = torch.randn(64, 3, 7, 7, requires_grad=True)
    torch.onnx.export(model, x, "torchToOnnx.onnx", verbose=False, input_names=['input'], output_names=['output'])

    # Load ONNX model
    onnx_model = onnx.load('/home/edgar/cougarvision/torchToOnnx.onnx')

    # Call the converter (input will be equal to the input_names parameter that you defined during exporting)
    k_model = onnx_to_keras(onnx_model, ['input'])
    return k_model
