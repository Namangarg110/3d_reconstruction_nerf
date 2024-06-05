# 3D Reconstruction using NeRF

## Project Overview

### Seamless Video Rendering: Drone View Synthesis
**Presented by:** Naman Garg, Mudit Jindal

**Course:** MSAI 490

### Project Goal
Achieve smooth video rendering without the need for costly setups. Our product leverages NeRF technology to produce high-quality videos efficiently and affordably.


## Neural Radiance Fields (NeRF)
NeRF leverages deep learning techniques to generate high-quality 3D models using unstructured 2D image data. It gets camera position for each frame using COLMAP (SfM, Bundle Adjustment).

### Key Features
- **High-Quality 3D Models**
- **Unstructured 2D Image Data Usage**
- **Camera Position Determination via COLMAP**

## Methodology

### Input Data
- **Shakespeare Garden Video:** Captured video frames.
- **COLMAP:** Determines camera positions.
- **Camera Motion Simulation:** Simulated camera movements.
- **Rendered Video:** Produces drone view perspectives.

### Neural Network Representation
- **Overfitting a Single Neural Network:** Focus on a single scene.
- **Input for Each Pixel:** Location (x, y, z) & viewing angle (θ, φ).
- **Ray Passing Through Scene:** Network outputs density (σ) and color (R, G, B).

### Training Dataset
- Each image pixel contributes data points (e.g., 512x512 pixels per image, 30 images, resulting in 7,864,320 data points).


### View Dependency
The neural radiance field model generates RGB colors based on spatial position and viewing direction, visualizing different camera angles.

### Positional Encoding
Used in Transformers to encode positional information and in NeRF to increase input dimensionality.

### Coarse and Fine Rendering Loss
- **Dual Network Optimization:** Optimize a coarse and a fine network simultaneously.
- **Usage:** The fine model is used at the end, while the coarse model penalizes the loss for better learning.

## Future Directions
- **Efficiency and Scalability:** Reduce computational costs using model pruning, quantization, and efficient network topologies.
- **Kolmogorov-Arnold Networks (KAN):** Explore for better 3D scene reconstruction and optimize training speed.
- **Learned Positional Encodings:** Implement to improve spatial information capture and reconstruction quality.
- **Out-of-View Area Filling:** Use diffusion models to create a more natural appearance by filling gaps.

## Repository Structure
- **.gitignore:** Initial commit
- **README.md:** Initial commit
- **render.py:** 3D rendering script
- **train.py:** Training script for 3D rendering
- **vanilla_nerf.py:** Basic implementation of NeRF
- **video_to_nerfstudio_dataset.py:** Converts video to NeRF studio dataset

For further details, refer to our [YouTube Video](https://youtu.be/LSVS7yeg644).

## Questions?
Feel free to reach out with any questions or clarifications.