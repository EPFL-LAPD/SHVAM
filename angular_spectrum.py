import jax
import jax.numpy as jnp
from jax import random, grad, jit

@jit
def angular_spectrum_propagation(field, z, wavelength, dx, n=1.0):
    """
    Propagates a 2D field using the angular spectrum method.

    This function is designed to be compatible with JAX's automatic differentiation.
    Based on bandlimited angular spectrum method from Matsushima et al. "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields" (2009).

    Args:
        field (jax.numpy.ndarray): Input field with shape (batch, 1, Nx, Ny).
        z (jax.numpy.ndarray): 1D array of propagation distances.
        wavelength (float): Wavelength of the light.
        dx (float): Pixel pitch (sampling interval) in the x and y dimensions.

    Keywords:
        n (float): Refractive index of the medium. Default is 1.0 (air).

    Returns:
        jax.numpy.ndarray: Propagated field with shape (batch, len(z), Nx, Ny).
    """

    # Adjust wavelength for refractive index
    wavelength = wavelength / n

    # Ensure the field has 4 dimensions
    if field.ndim != 4:
        raise ValueError(f"Input field must be 4D (batch, 1, Nx, Ny), but got {field.ndim}D")


    # not hard to generalize but just a bit more code to handle padding correctly
    if field.shape[-2] % 2 != 0 or field.shape[-1] % 2 != 0:
        raise ValueError("Input field spatial dimensions Nx and Ny must be even numbers for simplicity.")

    if field.shape[-1] != field.shape[-2]:
        raise ValueError("Input field spatial dimensions Nx and Ny must be equal for simplicity.")


    field = jnp.pad(field, ((0,0),(0,0),(field.shape[-2]//2, field.shape[-2]//2),(field.shape[-1]//2, field.shape[-1]//2)), mode='constant')

    # Get the spatial dimensions from the input field shape
    _, _, Nx, Ny = field.shape

    # Create spatial frequency coordinates
    fx = jnp.fft.fftfreq(Nx, d=dx)
    fy = jnp.fft.fftfreq(Ny, d=dx)
    Fx, Fy = jnp.meshgrid(fx, fy, indexing='ij')

    # --- Reshape to (1, 1, Nx, Ny) ---
    # Add two new axes at the beginning for broadcasting
    Fx_reshaped = Fx[jnp.newaxis, jnp.newaxis, :, :]
    Fy_reshaped = Fy[jnp.newaxis, jnp.newaxis, :, :]

    # Wave number
    k = 2 * jnp.pi / wavelength

    # Perform 2D FFT on the input field
    # axes=(-2, -1) applies FFT on the last two dimensions (Nx, Ny)
    A = jnp.fft.fft2(jnp.fft.ifftshift(field, axes=(-2,-1)), axes=(-2, -1))

    # 1, len(z), 1, 1
    z_vec = jnp.reshape(z, (1,-1,1, 1))

    # Calculate the propagation kernel in the frequency domain
    # This is the core of the angular spectrum method
    core = jnp.sqrt(0j + 1 - (wavelength * Fx)**2 - (wavelength * Fy)**2)

    delta_u = 1 / (Nx * dx)
    u_limit = 1 / (jnp.sqrt((2 * delta_u * z_vec)**2 + 1) * wavelength)
    W = jnp.logical_and(((Fy_reshaped**2 / u_limit**2 + 1 * Fx_reshaped**2 * wavelength**2) <= 1),
                        ((Fx_reshaped**2 / u_limit**2 + 1 * Fy_reshaped**2 * wavelength**2) <= 1))

    H = jnp.exp(1j * k * core * z_vec)


    U_z_freq = A * H * W

    # Inverse FFT to get the field in the spatial domain
    U_z_spatial = jnp.fft.fftshift(jnp.fft.ifft2(U_z_freq, axes=(-2, -1)),
                                   axes=(-2, -1))
    propagated_fields = U_z_spatial

    # Extract center Nx x Ny from 2*Nx x 2*Ny with proper even/odd handling
    Ny, Nx = propagated_fields.shape[-2:]  # Current dimensions (2*Ny, 2*Nx)
    target_Ny, target_Nx = Ny // 2, Nx // 2

    # Calculate start indices for center extraction
    start_y = (Ny - target_Ny) // 2
    start_x = (Nx - target_Nx) // 2


    # extract the center region
    propagated_fields = propagated_fields[:, :, start_y:start_y + target_Ny, start_x:start_x + target_Nx]

    return propagated_fields, H, W

