import torch
import torch.nn as nn
import json
import numpy as np

# Define the model
class EchoImageVerificationModel(nn.Module):
    def __init__(self, image_shape, challenge_audio_shape, recorded_audio_shape):
        """
        image_shape: (channels, H, W), e.g. (1, 16, 16)
        challenge_audio_shape: (channels, length), e.g. (1, 1000)
        recorded_audio_shape: (channels, length), e.g. (1, 1000)
        """
        super(EchoImageVerificationModel, self).__init__()
        
        # Save the original shapes for reshaping later.
        self.image_shape = image_shape
        self.challenge_audio_shape = challenge_audio_shape
        self.recorded_audio_shape = recorded_audio_shape
        
        # --- Audio branch ---
        # A single Conv1d layer with fewer channels and a modest kernel,
        # followed by an AdaptiveAvgPool1d to produce an output of shape (batch, 4, 1).
        self.audio_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Output: (batch, 4, 1)
        )
        
        # --- Image branch ---
        # Two convolutional layers with pooling to shrink dimensions.
        # The first conv keeps the spatial dimensions (with padding) then a MaxPool2d halves them.
        # The second conv (with stride 2) further reduces the resolution.
        # Finally, an AdaptiveAvgPool2d produces an output of shape (batch, 8, 1, 1).
        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),  # (batch, 4, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                       # (batch, 4, 8, 8)
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),  # (batch, 8, 4, 4)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)                                                     # (batch, 8, 1, 1)
        )
        
        # --- Fully Connected Block ---
        # With an 8-element image feature and 4-element features from each audio branch,
        # the total concatenated feature vector has size 8 + 4 + 4 = 16.
        self.fc = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image_flat, challenge_audio_flat, recorded_audio_flat):
        """
        Each input is a flattened tensor of shape (batch, num_elements).
        We reshape them back to the original shapes before processing.
        """
        # Reshape the flattened inputs back to their original dimensions.
        # Here, "-1" preserves the batch dimension.
        image = image_flat.view(-1, *self.image_shape)
        challenge_audio = challenge_audio_flat.view(-1, *self.challenge_audio_shape)
        recorded_audio = recorded_audio_flat.view(-1, *self.recorded_audio_shape)

        # Process the audio inputs.
        challenge_feat = self.audio_conv(challenge_audio).flatten(start_dim=1)
        recorded_feat = self.audio_conv(recorded_audio).flatten(start_dim=1)

        # Process the image input.
        image_feat = self.image_conv(image).flatten(start_dim=1)

        # Combine all features.
        fused_features = torch.cat((image_feat, challenge_feat, recorded_feat), dim=1)

        # Decision output.
        output = self.fc(fused_features)
        return output

# -------------------------------------------------------------------
# Define the original input shapes (excluding batch dimension).
# (For the JSON you saved and the actual ONNX export, we assume the data was flattened.)
image_shape = (1, 16, 16)           # Grayscale image: (channels, H, W)
challenge_audio_shape = (1, 1000)   # Challenge audio: (channels, length)
recorded_audio_shape = (1, 1000)    # Recorded audio: (channels, length)

# Create the model and set it to evaluation mode.
model = EchoImageVerificationModel(image_shape, challenge_audio_shape, recorded_audio_shape)
model.eval()

# -------------------------------------------------------------------
# Create dummy inputs.
# For ONNX export we need tensor inputs.
# First, generate dummy tensors with the original shape and then flatten them.
dummy_image_tensor = torch.randn(1, *image_shape).view(1, -1)
dummy_challenge_audio_tensor = torch.randn(1, *challenge_audio_shape).view(1, -1)
dummy_recorded_audio_tensor = torch.randn(1, *recorded_audio_shape).view(1, -1)

# For saving JSON, we can convert these tensors to flattened lists.
dummy_image = dummy_image_tensor.detach().cpu().numpy().flatten().tolist()
dummy_challenge_audio = dummy_challenge_audio_tensor.detach().cpu().numpy().flatten().tolist()
dummy_recorded_audio = dummy_recorded_audio_tensor.detach().cpu().numpy().flatten().tolist()

# -------------------------------------------------------------------
# Export the model to ONNX.
torch.onnx.export(
    model,
    (dummy_image_tensor, dummy_challenge_audio_tensor, dummy_recorded_audio_tensor),  # Provide a tuple of tensor inputs.
    "echo_image_verification.onnx",
    export_params=True,        # Store the trained parameter weights inside the model file.
    opset_version=10,          # The ONNX version to export the model to.
    do_constant_folding=True,  # Whether to execute constant folding for optimization.
    input_names=['image_flat', 'challenge_audio_flat', 'recorded_audio_flat'],  # The model's input names.
    output_names=['output'],   # The model's output names.
)

print("✅ ONNX Model saved as 'echo_image_verification.onnx' with flattened inputs.")

# -------------------------------------------------------------------
# Create a dummy JSON input file.
# This JSON file contains a key "input_data" holding three flattened lists.
dummy_input_dict = {
    "input_data": [
        dummy_image, 
        dummy_challenge_audio, 
        dummy_recorded_audio
    ],
}

input_filename = "echo_image_verification_input.json"
with open(input_filename, "w") as json_file:
    json.dump(dummy_input_dict, json_file, indent=4)

print(f"✅ Dummy input JSON saved as '{input_filename}'")