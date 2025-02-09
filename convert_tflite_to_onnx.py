import tensorflow as tf
import tf2onnx
import onnx
from onnx import helper, checker, shape_inference

# Define file paths
tflite_model_path = "lite-model_yamnet_classification_tflite_1.tflite"
onnx_cleaned_model_path = "yamnet-small/source.onnx"

# Load the TFLite model to get input details
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()

# Extract actual input name
actual_input_name = input_details[0]['name']
print(f"✅ TFLite input name detected: {actual_input_name}")

# Convert TFLite to ONNX with correct input name
onnx_model, _ = tf2onnx.convert.from_tflite(
    tflite_path=tflite_model_path, 
    opset=13
)

# Verify model structure
checker.check_model(onnx_model)

# Rename the first input to "input"
graph = onnx_model.graph
graph.input[0].name = "input"

# Ensure all references are updated properly
for node in graph.node:
    for i, inp in enumerate(node.input):
        if inp == actual_input_name:
            node.input[i] = "input"

# Infer shapes to fix missing shape issues
onnx_model = shape_inference.infer_shapes(onnx_model)

# Save the cleaned ONNX model
onnx.save(onnx_model, onnx_cleaned_model_path)
print(f"✅ Successfully converted and cleaned ONNX model: {onnx_cleaned_model_path}")

# Final validation check
checker.check_model(onnx.load(onnx_cleaned_model_path))
print("✅ ONNX model passed final validation!")