# python -m tf2onnx.convert --tflite /Users/chenweijia/Documents/code/NetTCR-2.2/models/nettcr_2_2_pan/checkpoint/t.0.v.1.tflite --output /Users/chenweijia/Documents/code/nettcr_pytorch/models/t.0.v.1.onnx --opset 13
import torch
import onnx
from onnx2torch import convert

input_dir = "models/onnx/"
input_filename = "t.0.v.1.onnx"
output_dir = "models/pt/"
output_filename = "t.0.v.1.pth"

# Convert ONNX to PyTorch
pytorch_model = convert(input_dir + input_filename)

# Save the PyTorch model
torch.save(pytorch_model.state_dict(), output_dir + output_filename)