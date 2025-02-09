import torch
import torchvision.models as models
import onnx
import numpy as np
import json
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import onnxruntime as ort
import os
from onnx import helper, numpy_helper

# --- Define Calibration Data Reader ---
class DataReader(CalibrationDataReader):
    def __init__(self, sample_inputs):
        self.sample_inputs = sample_inputs
        self.data_iter = iter([{"input": sample_inputs}])

    def get_next(self):
        return next(self.data_iter, None)

    def rewind(self):
        self.data_iter = iter([{"input": self.sample_inputs}])

# --- Function to Convert PyTorch Model to ONNX ---
def convert_pytorch_to_onnx(model, input_shape, onnx_path):
    model.eval()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11  # Ensure latest ONNX opset for best quantization support
    )

# --- Function to Fix Quantization Nodes ---
def fix_quantization_nodes(model_path, output_path):
    model = onnx.load(model_path)
    graph = model.graph

    new_nodes = []

    for node in graph.node:
        if node.op_type == "QuantizeLinear":
            scale_name = node.input[1]
            zero_point_name = node.input[2]

            new_nodes.append(helper.make_node(
                "QuantizeLinear",
                inputs=[node.input[0], scale_name, zero_point_name],
                outputs=node.output,
                name=node.name
            ))

        elif node.op_type == "DequantizeLinear":
            scale_name = node.input[1]
            zero_point_name = node.input[2]

            new_nodes.append(helper.make_node(
                "DequantizeLinear",
                inputs=[node.input[0], scale_name, zero_point_name],
                outputs=node.output,
                name=node.name
            ))
        else:
            new_nodes.append(node)

    graph.ClearField("node")
    graph.node.extend(new_nodes)
    onnx.save(model, output_path)
    print(f"✅ Fixed quantization model saved as {output_path}")

# --- Function to Quantize ONNX Model (Static UINT8) ---
def quantize_onnx_model(onnx_model_path, quantized_model_path, sample_input):
    data_reader = DataReader(sample_input)
    quantize_static(
        onnx_model_path,
        quantized_model_path,
        data_reader,
        weight_type=QuantType.QUInt8,  # Ensure weights are uint8
        activation_type=QuantType.QUInt8,  # Ensure activations are uint8
    )
    return quantized_model_path

# ### **1. ShuffleNetV2 - Image Model**
# Load pre-trained ShuffleNetV2
shufflenet = models.shufflenet_v2_x0_5(pretrained=True)  # Smallest ShuffleNet variant
# shufflenet.fc = torch.nn.Identity()  # Remove classification head

# Convert to ONNX
onnx_path_shufflenet = "shufflenetv2.onnx"
convert_pytorch_to_onnx(shufflenet, (1, 3, 224, 224), onnx_path_shufflenet)

# Generate Calibration Data (Sample Input)
calibration_sample = np.random.rand(1, 3, 224, 224).astype(np.float32)

# Quantize to Static UINT8
quantized_shufflenet_path = "shufflenetv2_quantized_uint8.onnx"
quantize_onnx_model(onnx_path_shufflenet, quantized_shufflenet_path, calibration_sample)

# Fix Quantization Nodes for EZKL Compatibility
fixed_quantized_model_path = "shufflenetv2_fixed_quantized_uint8.onnx"
fix_quantization_nodes(quantized_shufflenet_path, fixed_quantized_model_path)

print("✅ ShuffleNetV2 successfully quantized and fixed for EZKL compatibility!")
