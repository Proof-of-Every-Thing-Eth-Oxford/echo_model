import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchaudio
import torchvision.transforms as transforms

# -------------------------------
# Custom Dataset Definition
# -------------------------------
class EchoDataset(Dataset):
    def __init__(self, root_dir, transform_image=None, target_sample_rate=8000, duration=1.0):
        """
        Args:
            root_dir (str): Root directory containing one folder per sample.
            transform_image (callable, optional): Transform to be applied on the image.
            target_sample_rate (int): Desired audio sample rate.
            duration (float): Duration in seconds of the audio clips.
        """
        self.root_dir = root_dir
        self.transform_image = transform_image if transform_image is not None else transforms.ToTensor()
        self.target_sample_rate = target_sample_rate
        self.audio_length = int(target_sample_rate * duration)
        self.samples = []
        
        # Assume each subfolder of root_dir is a sample.
        for sample_folder in os.listdir(root_dir):
            sample_path = os.path.join(root_dir, sample_folder)
            if os.path.isdir(sample_path):
                image_path = os.path.join(sample_path, "image.png")  # or "image.jpg" as needed
                challenge_audio_path = os.path.join(sample_path, "challenge_sound.wav")
                recorded_audio_path = os.path.join(sample_path, "recorded_sound.wav")
                label_path = os.path.join(sample_path, "label.txt")  # Optional label file
                
                if (os.path.exists(image_path) and os.path.exists(challenge_audio_path)
                        and os.path.exists(recorded_audio_path)):
                    # If a label file exists, read the label; otherwise, use a dummy label (e.g. 1.0)
                    if os.path.exists(label_path):
                        with open(label_path, "r") as f:
                            label = float(f.read().strip())
                    else:
                        label = 1.0
                    self.samples.append((image_path, challenge_audio_path, recorded_audio_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, challenge_audio_path, recorded_audio_path, label = self.samples[idx]
        
        # Load image and convert to grayscale; resize to 64x64
        image = Image.open(image_path).convert("L")
        if self.transform_image:
            image = self.transform_image(image)  # Expected shape: (1, 64, 64)
        else:
            image = transforms.ToTensor()(image)
        
        # Load audio files using torchaudio
        challenge_audio, sr_challenge = torchaudio.load(challenge_audio_path)  # shape: (channels, samples)
        recorded_audio, sr_recorded = torchaudio.load(recorded_audio_path)
        
        # Resample audio if needed
        if sr_challenge != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr_challenge, new_freq=self.target_sample_rate)
            challenge_audio = resampler(challenge_audio)
        if sr_recorded != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr_recorded, new_freq=self.target_sample_rate)
            recorded_audio = resampler(recorded_audio)
        
        # Ensure each audio tensor is exactly self.audio_length samples long.
        # (Assumes audio tensors have shape (1, samples).)
        def fix_audio(audio):
            if audio.size(1) > self.audio_length:
                audio = audio[:, :self.audio_length]
            elif audio.size(1) < self.audio_length:
                pad_amount = self.audio_length - audio.size(1)
                audio = torch.nn.functional.pad(audio, (0, pad_amount))
            return audio
        
        challenge_audio = fix_audio(challenge_audio)
        recorded_audio = fix_audio(recorded_audio)
        
        # Convert label to a tensor
        label = torch.tensor([label], dtype=torch.float32)
        
        return image, challenge_audio, recorded_audio, label

# -------------------------------
# Instantiate Dataset and DataLoader
# -------------------------------
data_dir = "data"  # Root directory containing sample subfolders
transform_img = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
dataset = EchoDataset(root_dir=data_dir, transform_image=transform_img)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# -------------------------------
# Define the Model (Assumed already defined)
# -------------------------------
# Here we assume that EchoImageVerificationModel is already defined as in your previous code.
# For example, the model expects:
#   - image input: shape (batch, 1, 64, 64)
#   - challenge audio: shape (batch, 1, 8000)
#   - recorded audio: shape (batch, 1, 8000)
model = EchoImageVerificationModel()
model.train()

# -------------------------------
# Training Setup
# -------------------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# Training Loop
# -------------------------------
for epoch in range(10):
    epoch_loss = 0.0
    for images, challenge_sounds, recorded_sounds, labels in dataloader:
        optimizer.zero_grad()
        
        # Forward pass (using real loaded data)
        outputs = model(images, challenge_sounds, recorded_sounds)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")