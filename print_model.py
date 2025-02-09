import argparse
from collections import Counter
import onnx

def count_onnx_node_types(model_path):
    model = onnx.load(model_path)
    node_types = [node.op_type for node in model.graph.node]
    node_counts = Counter(node_types)

    print("🔹 ONNX Model Node Type Counts:")
    for node_type, count in node_counts.items():
        print(f"- {node_type}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count ONNX model node types.")
    parser.add_argument("model_path", type=str, help="Path to the ONNX model.")
    args = parser.parse_args()
    
    count_onnx_node_types(args.model_path)