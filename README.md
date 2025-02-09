# Echo-Based Reality to Image Verification

> **Disclaimer**: This work was conducted by **Security Enthusiasts** with minimal advanced Machine Learning background. The focus was on building a proof-of-concept for echo-based image verification rather than optimizing for state-of-the-art performance or best practices in ML engineering.

This repository contains our experiments and scripts to build, train, and export an **audio+image verification model**. The idea is to combine three inputs:

1. **A challenge audio** (the known reference sound),
2. **A recorded audio** (the actual sound recorded in real-time, potentially containing an echo),
3. **An image** (e.g., a face or an object in front of the device).

The model outputs a **single probability** indicating whether:
1. The recorded audio **matches** the challenge audio (i.e., the echo’s sound is indeed from that challenge), **and**
2. The echo’s characteristics **match** the depth profile implied by the image (face/other object dimensions, etc.).

In essence, the model attempts to validate:
- **Was the object truly in front of the device?**  
- **Did the recorded echo come from the correct challenge sound?**


## Model Overview

Our final “echo” model is shown below. It takes **three inputs**:
1. **Challenge audio** (1D conv branch)
2. **Recorded audio** (1D conv branch)
3. **Image** (2D conv branch)

All branches produce feature vectors which are **concatenated** into a single fused tensor. A series of **fully connected layers** outputs a single probability value.

**ASCII Sketch** of the pipeline:

```
                               ┌───────────────────────────┐
                               │ [Conv1D + pooling]        │
 [Challenge Audio]  ──────────▶│     Output feat (C_feat)  │
                               └───────────────────────────┘
                                                    \
                                                     \
 [Recorded Audio]   ──────────▶(same 1D conv branch)─> feat (R_feat)
                                                      \
                                                       \
                                                        ┌────────────────────┐
 [Image]            ──────────────────────────────────▶ │ [Conv2D + pooling] │
                                                        │    Output feat (I_feat)
                                                        └────────────────────┘

  (C_feat, R_feat, I_feat) -- Concatenate --> [Fully Connected Layer(s)] --> [ Sigmoid ] --> Probability
```

### How It Works

1. **Inputs**  
   - **Challenge sound**: The known reference audio we expect to hear.  
   - **Recorded sound**: The actual captured audio in real time, which should contain an **echo** if a real object is present.  
   - **Image**: An image (e.g., of a face or an object) that we want to verify is physically in front of the device.

2. **Convolutions & Feature Extraction**  
   - Each audio input (challenge and recorded) runs through 1D convolutions and pooling, extracting relevant audio features.  
   - The image runs through 2D convolutions and pooling, extracting spatial features.

3. **Fusion**  
   - The three feature vectors are concatenated into one **fused feature vector**.

4. **Decision**  
   - A final fully connected (FC) block outputs **1 probability** ∈ (0, 1).  
   - **High** probability = “Yes, the recorded audio matches the challenge echo, and the echo’s depth profile matches the image.”  
   - **Low** probability = “No, it’s likely not the correct echo and/or not the correct object.”


## Repository Contents

Below is an overview of the files/folders in this repository:

| File/Folder                | Description                                                                                                     |
|----------------------------|-----------------------------------------------------------------------------------------------------------------|
| **convert_tflite_to_onnx.py** | Script to convert TensorFlow Lite models into ONNX format. Useful if you have TFLite models you want to port over. |
| **onnx_env/**                | A Python virtual environment directory for ONNX-related dependencies. <br>**Recommended**: add to `.gitignore`.  |
| **print_model.py**           | Utility script to print the internal structure (nodes, layers) of an ONNX file. Helpful for debugging.          |
| **uint8_quant_model.py**     | Attempts to quantize pretrained models to `uint8` in order to reduce size. Ultimately not successful for our case. |
| **custom/**                  | Folder containing our **custom model** code, training scripts, ONNX exports, and some proving artifacts.         |
| **prep_shufflenetv2_model.py** | Script exploring **ShuffleNetV2** for image recognition usage. Not the final approach but was part of the experimentation. |
| **split_onnx.py**            | Splits a large ONNX file into smaller sub-ONNX files in an attempt to optimize SNARK proving. Inspired by [ezkl/pull/855](https://github.com/zkonduit/ezkl/pull/855). |

### Inside `custom/`

| File/Folder                     | Description                                                                                                                                           |
|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **echo_image_verification.onnx**      | The **final ONNX model** used for echo-based image verification.                                                                                     |
| **echo_image_verification_input.json** | A **sample input** JSON file demonstrating how to feed challenge audio, recorded audio, and an image into the model (as flattened arrays).           |
| **training.py**                       | The **training script** that defines and trains a PyTorch model before exporting to ONNX.                                                           |
| **simple_echolot.py**                 | Defines the architecture (layers, forward pass) for the custom echo-based model.                                                                     |
| **split/**                            | Contains **split ONNX models** (sub-models) if you attempted to break the network down for separate proving steps.                                   |

## Usage Notes

- Most scripts here were prototypes. Production-level code or advanced ML techniques were not our priority.
- The final custom approach is in `custom/`.
- For any onnx-specific usage, see `convert_tflite_to_onnx.py`, `print_model.py`, and `split_onnx.py`.
- If using [ezkl](https://github.com/zkonduit/ezkl) or other ZK frameworks, you may need the `pk.key`, `vk.key`, `witness.json`, etc. from the `custom` folder.

---

We hope this repository serves as a **starting point** for anyone interested in echo-based verification. **Security Enthusiasts** built this prototype to demonstrate the concept of verifying an object’s presence with a matching echo.
