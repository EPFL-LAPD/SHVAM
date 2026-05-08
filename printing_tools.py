import matplotlib.pyplot as plt
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax import random, grad, jit
import numpy as np




def make_hollow_box(cube_size, cube_size_z, edge_width, N_target, dx, z_distances):
    x = jnp.linspace(-N_target/2, N_target/2, N_target) * dx
    y = jnp.linspace(-N_target/2, N_target/2, N_target) * dx
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    X = X.reshape(1, 1, N_target, N_target)
    Y = Y.reshape(1, 1, N_target, N_target)
    Z = z_distances.reshape(1, z_distances.shape[0], 1, 1)

    # Define 500µm cube centered at origin
    half_size = cube_size / 2
    half_width = edge_width / 2

    solid_cube = (jnp.abs(X) < cube_size / 2) *(jnp.abs(Y) < cube_size / 2) * (jnp.abs(Z) < cube_size_z / 2)
    subtract_cube_1 = (jnp.abs(X) < cube_size / 2 - edge_width) * (jnp.abs(Y) < cube_size / 2 - edge_width) * 1.0
    subtract_cube_2 = (jnp.abs(Z) < cube_size_z / 2 - edge_width) * (jnp.abs(Y) < cube_size / 2 - edge_width) * 1.0
    subtract_cube_3 = (jnp.abs(Z) < cube_size_z / 2 - edge_width) * (jnp.abs(X) < cube_size / 2 - edge_width) * 1.0

    hollow_box = ((solid_cube - subtract_cube_1 - subtract_cube_2 - subtract_cube_3) > 0) * 1.0

    return hollow_box


def make_hollow_box2(cube_size, cube_size_z, edge_width, N_target, dx, z_distances):
    x = jnp.linspace(-N_target/2, N_target/2, N_target) * dx
    y = jnp.linspace(-N_target/2, N_target/2, N_target) * dx
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    X = X.reshape(1, 1, N_target, N_target)
    Y = Y.reshape(1, 1, N_target, N_target)
    Z = z_distances.reshape(1, z_distances.shape[0], 1, 1)

    z_max = jnp.max(Z)

    # Define 500µm cube centered at origin
    half_size = cube_size / 2
    half_width = edge_width / 2

    solid_cube = (jnp.abs(X) < cube_size / 2) *(jnp.abs(Y) < cube_size / 2) * (jnp.abs(Z - z_max / 2) < cube_size_z / 2)
    subtract_cube_1 = (jnp.abs(X) < cube_size / 2 - edge_width) * (jnp.abs(Y) < cube_size / 2 - edge_width) * 1.0
    subtract_cube_2 = (jnp.abs(Z-z_max / 2) < cube_size_z / 2 - edge_width * 2) * (jnp.abs(Y) < cube_size / 2 - edge_width) * 1.0
    subtract_cube_3 = (jnp.abs(Z-z_max / 2) < cube_size_z / 2 - edge_width * 2) * (jnp.abs(X) < cube_size / 2 - edge_width) * 1.0

    hollow_box = ((solid_cube - subtract_cube_1 - subtract_cube_2 - subtract_cube_3) > 0) * 1.0

    return hollow_box




@jit
def calculate_iou(prediction, target, threshold):
    """
    Calculates the Intersection over Union (IoU) for JAX arrays.

    Args:
        prediction: The predicted output array with continuous values.
        target: The ground truth binary mask (0s and 1s).
        threshold: The value to binarize the prediction array.

    Returns:
        The IoU score as a scalar float.
    """
    # 1. Binarize the prediction based on the threshold
    predicted_mask = (prediction >= threshold).astype(jnp.float32)

    # 2. Ensure the target is also treated as a binary mask
    target_mask = target.astype(jnp.float32)

    # 3. Calculate intersection: The area where both masks are active (1)
    intersection = jnp.sum(predicted_mask * target_mask)

    # 4. Calculate union: The total area covered by either mask
    # Union(A, B) = A + B - Intersection(A, B)
    union = jnp.sum(predicted_mask) + jnp.sum(target_mask) - intersection

    # 5. Compute IoU, adding a small epsilon to avoid division by zero
    iou = intersection / (union + 1e-8)

    return iou


# save as .npz
def save_patterns(filename, patterns):
    """
    Saves patterns to a .npz file.

    Args:
        filename: The name of the file to save the patterns to.
        patterns: A list or array of patterns to save.
    """
    np.savez_compressed(filename, patterns)
    print(f"Patterns saved to {filename}")
    return 0


def load_patterns(filename):
    """
    Loads patterns from a .npz file.

    Args:
        filename: The name of the file to load the patterns from.

    Returns:
        A list of loaded patterns.
    """
    data = jnp.load(filename)
    patterns = data["patterns"]
    print(f"Patterns loaded from {filename}")
    return patterns


import numpy as np
import jax.numpy as jnp
from PIL import Image
import os

def save_patterns(arr, bitmap_path, npy_filename):
    arr = jnp.fft.fftshift(arr, axes=(-2,-1))
    arr = arr.squeeze(1)

    pad_width = 1920 - 1200
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Pad array: (batch, height, width)
    phase = np.pad(arr, ((0, 0), (0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)

    # Calculate phase and normalize to 0-255
    phase = jnp.angle(jnp.exp(1j * phase)) + jnp.pi
    phase_image = phase / (2 * np.pi) * 255

    # Convert to numpy
    phase_np = np.array(phase)

    # Create directory for npy file
    npy_dir = os.path.dirname(npy_filename)
    if npy_dir:
        os.makedirs(npy_dir, exist_ok=True)

    # Save as .npy
    np.save(npy_filename, phase_np)



    # Create directory and save as 20 individual 8-bit bitmaps
    os.makedirs(bitmap_path, exist_ok=True)
    for i in range(arr.shape[0]):
        img = Image.fromarray(np.array(phase_image)[i], mode='L')
        img.save(os.path.join(bitmap_path, f'pattern_{i:02d}.bmp'))

