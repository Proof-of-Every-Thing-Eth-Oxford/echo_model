import torch
import torchvision.models as models
import torch.nn as nn
import onnx
import json
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType

# --- Function to Convert PyTorch Model to ONNX ---
def convert_pytorch_to_onnx(model, input_shape, onnx_path):
    """Converts a PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )

# --- Function to Quantize ONNX Model ---
def quantize_onnx_model(onnx_model_path, quantized_model_path):
    """Applies dynamic INT8 quantization on an ONNX model."""
    quantize_dynamic(onnx_model_path, quantized_model_path, weight_type=QuantType.QInt8)
    return quantized_model_path

# ### **1. ShuffleNetV2 - Image Model**
# Load pre-trained ShuffleNetV2
shufflenet = models.shufflenet_v2_x0_5(pretrained=True)  # Using the smallest ShuffleNetV2 (0.5x)
shufflenet.fc = nn.Identity()  # Remove classification layer

# Convert to ONNX
onnx_path_shufflenet = "shufflenetv2.onnx"
convert_pytorch_to_onnx(shufflenet, (1, 3, 224, 224), onnx_path_shufflenet)

# Quantize to INT8
quantized_shufflenet_path = "shufflenetv2_quantized.onnx"
quantize_onnx_model(onnx_path_shufflenet, quantized_shufflenet_path)

# --- Generate Sample Input JSON ---
image_input_sample = {
    "input": np.random.rand(1, 3, 224, 224).tolist()  # Random normalized image tensor
}
with open("shufflenetv2_sample_input.json", "w") as f:
    json.dump(image_input_sample, f, indent=4)

print("✅ ShuffleNetV2 converted & quantized successfully!")
print("✅ Sample input JSON file generated!")