import jax.numpy as jnp
import jax
from jax import random, grad, jit
from angular_spectrum import angular_spectrum_propagation
import optax
from functools import partial
from typing import NamedTuple
import chex


@jit
def conv(array, kernel):
    """
    Convolve an array with a kernel using FFT-based convolution.

    Args:
        array: Input array to convolve.
        kernel: Convolution kernel (centered format); will be shifted internally.

    Returns:
        Absolute value of the convolution result, same shape as `array`.
    """
    kernel_shifted = jnp.fft.ifftshift(kernel, axes=(1, 2, 3))

    array_fft = jnp.fft.fftn(array, axes=(1, 2, 3))
    kernel_fft = jnp.fft.fftn(kernel_shifted, axes=(1, 2, 3))

    result_fft = array_fft * kernel_fft
    result = jnp.fft.ifftn(result_fft, axes=(1, 2, 3))

    return jnp.abs(result)


@partial(jit, static_argnames=['n_time_steps'])
def forward_print_diffusion(
    phase_patterns,
    z,
    wavelength,
    dx,
    target,
    beam_slm,
    beam_centered,
    absorption_decay,
    intensity_scaling,
    n_medium,
    diffusion_coefficient_oxygen,
    diffusion_coefficient_tempo,
    time,
    n_time_steps,
    initial_concentration_oxygen,
    initial_concentration_tempo,
):
    """
    Simulate volumetric photopolymerisation printing with two diffusing inhibitors.

    The function propagates a set of SLM phase patterns to compute the accumulated
    light intensity inside the resin volume, then iterates a reaction-diffusion model
    over `n_time_steps` steps.  At each step:

    1. Radicals are generated proportional to the local intensity.
    2. Oxygen inhibitor quenches radicals first, then TEMPO inhibitor quenches the
       remainder.
    3. Surviving radicals are accumulated into the dose field.
    4. Each field (oxygen, TEMPO, dose) diffuses via Gaussian convolution.

    Args:
        phase_patterns: SLM phase patterns, shape ``(N_patterns, 1, Nx, Ny)``.
        z: Propagation distances along the optical axis, shape ``(Nz,)``.
        wavelength: Illumination wavelength (metres).
        dx: Lateral pixel pitch (metres).
        target: Target intensity volume, shape ``(1, Nz, Nx, Ny)``.
        beam_slm: Complex field at the SLM plane, shape ``(1, 1, Nx_slm, Ny_slm)``.
        beam_centered: Apodisation / beam-profile field in the target volume,
            shape ``(1, 1, Nx, Ny)``.
        absorption_decay: Axial absorption envelope, broadcastable to the volume.
        intensity_scaling: Scalar factor applied to the summed intensity.
        n_medium: Refractive index of the printing medium.
        diffusion_coefficient_oxygen: Diffusion coefficient of the oxygen inhibitor
            (m² s⁻¹).
        diffusion_coefficient_tempo: Diffusion coefficient of the TEMPO inhibitor
            (m² s⁻¹).
        time: Total exposure / simulation time (seconds).
        n_time_steps: Number of discrete time steps (static integer).
        initial_concentration_oxygen: Initial uniform concentration of the oxygen
            inhibitor (arbitrary units), scalar.
        initial_concentration_tempo: Initial uniform concentration of the TEMPO
            inhibitor (arbitrary units), scalar.

    Returns:
        dose: Accumulated radical dose volume, shape ``(1, Nz, Nx, Ny)``.
        inhibitor_oxygen: Residual oxygen concentration after exposure,
            shape ``(1, Nz, Nx, Ny)``.
        inhibitor_tempo: Residual TEMPO concentration after exposure,
            shape ``(1, Nz, Nx, Ny)``.
        intensity: Time-integrated light intensity volume, shape ``(1, Nz, Nx, Ny)``.
    """
    N_target = target.shape[-1]

    # Build spatial coordinate grids for the diffusion kernels
    x = jnp.linspace(-N_target / 2, N_target / 2, N_target, endpoint=False) * dx
    y = jnp.linspace(-N_target / 2, N_target / 2, N_target, endpoint=False) * dx
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    X = X.reshape(1, 1, N_target, N_target)
    Y = Y.reshape(1, 1, N_target, N_target)

    zmin = jnp.min(z)
    zmax = jnp.max(z) - zmin
    z_centered = jnp.linspace(-zmax / 2, zmax / 2, z.shape[0], endpoint=False)
    Z = z_centered.reshape(1, z.shape[0], 1, 1)

    r2 = X**2 + Y**2 + Z**2
    delta_t = time / n_time_steps

    # Gaussian diffusion kernels for each species
    kernel_oxygen = jnp.exp(-r2 / (4 * diffusion_coefficient_oxygen * delta_t))
    kernel_oxygen = kernel_oxygen / jnp.sum(kernel_oxygen)

    kernel_tempo = jnp.exp(-r2 / (4 * diffusion_coefficient_tempo * delta_t))
    kernel_tempo = kernel_tempo / jnp.sum(kernel_tempo)


    # --- Optical propagation ---

    def propagate_and_accumulate(accumulated_field, phase_pattern):
        """Propagate one phase pattern and add its intensity to the accumulator."""
        if phase_pattern.ndim == 3:
            phase_pattern = phase_pattern[:, jnp.newaxis, :, :]  # (1, 1, Nx, Ny)

        field = jnp.fft.fftshift(
            jnp.fft.fft2(beam_slm * jnp.exp(1j * phase_pattern), axes=(-2, -1)),
            axes=(-2, -1),
        )
        field_crop = field[:, :, 0:N_target, 0:N_target]
        propagated_intensity = (
            beam_centered[:, :, 0:N_target, 0:N_target]
            * absorption_decay
            * jnp.abs(angular_spectrum_propagation(field_crop, z, wavelength, dx, n=n_medium)[0]) ** 2
        )
        return accumulated_field + propagated_intensity, None

    init_accumulated = jnp.zeros((1, len(z), N_target, N_target), dtype=jnp.float32)
    summed_fields, _ = jax.lax.scan(
        propagate_and_accumulate, init_accumulated, phase_patterns
    )
    summed_fields *= (1 / phase_patterns.shape[0]) * intensity_scaling
    intensity = summed_fields

    # --- Reaction-diffusion time loop ---


    def body_fn(carry, _):
        inhibitor_O2, inhibitor_tempo, dose = carry
    
        radicals = intensity
        radicals_n = jnp.maximum(0, radicals - inhibitor_O2)
        inhibitor_O2 = jnp.maximum(0, inhibitor_O2 - radicals)
        radicals = radicals_n
        
        radicals_n = jnp.maximum(0, radicals - inhibitor_tempo)
        inhibitor_tempo = jnp.maximum(0, inhibitor_tempo - radicals)
        radicals = radicals_n


        dose = dose + radicals
    
        inhibitor_O2 = conv(inhibitor_O2, kernel_oxygen)
        inhibitor_tempo = conv(inhibitor_tempo, kernel_tempo)
    
        return (inhibitor_O2, inhibitor_tempo, dose), None  # None = don't collect outputs



    inhibitor_oxygen = initial_concentration_oxygen + 0 * summed_fields
    inhibitor_tempo = initial_concentration_tempo + 0 * summed_fields
    dose = jnp.zeros_like(inhibitor_oxygen)

    (inhibitor_oxygen, inhibitor_tempo, dose), _ = jax.lax.scan(
        body_fn, (inhibitor_oxygen, inhibitor_tempo, dose), None, length=n_time_steps
    )

    return dose, inhibitor_oxygen, inhibitor_tempo, intensity


def optimize_print(phases, fun, max_iter=100, tol=1e-3):
    """
    Optimise SLM phase patterns using the L-BFGS algorithm via Optax.

    Args:
        phases: Initial phase parameters (JAX array).
        fun: Scalar-valued loss function of the phases.
        max_iter: Maximum number of L-BFGS iterations.
        tol: Convergence tolerance on the gradient norm.

    Returns:
        phases_final: Optimised phase parameters.
        losses: List of loss values recorded at each iteration.
    """
    losses = []

    def run_opt(init_params, fun, opt, max_iter, tol):
        """Run the L-BFGS while-loop and return final parameters and state."""
        value_and_grad_fun = optax.value_and_grad_from_state(fun)

        def step(carry):
            params, state = carry
            value, grad = value_and_grad_fun(params, state=state)
            updates, state = opt.update(
                grad, state, params, value=value, grad=grad, value_fn=fun
            )
            params = optax.apply_updates(params, updates)
            return params, state

        def continuing_criterion(carry):
            _, state = carry
            iter_num = optax.tree.get(state, 'count')
            grad = optax.tree.get(state, 'grad')
            err = optax.tree.norm(grad)
            return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

        init_carry = (init_params, opt.init(init_params))
        final_params, final_state = jax.lax.while_loop(
            continuing_criterion, step, init_carry
        )
        return final_params, final_state

    class InfoState(NamedTuple):
        """Carries the iteration counter for the logging gradient transformation."""
        iter_num: chex.Numeric

    def print_info():
        """
        Optax gradient transformation that logs loss and gradient norm each iteration.

        Returns:
            An ``optax.GradientTransformationExtraArgs`` that passes updates through
            unchanged while recording diagnostics as a side effect.
        """
        def init_fn(params):
            del params
            return InfoState(iter_num=0)

        def update_fn(updates, state, params, *, value, grad, **extra_args):
            del params, extra_args
            jax.experimental.io_callback(lambda v: losses.append(v), None, value)
            jax.debug.print(
                'Iteration: {i}, Value: {v}, Gradient norm: {e}',
                i=state.iter_num,
                v=value,
                e=optax.tree.norm(grad),
            )
            return updates, InfoState(iter_num=state.iter_num + 1)

        return optax.GradientTransformationExtraArgs(init_fn, update_fn)

    opt = optax.chain(print_info(), optax.lbfgs())

    print(
        f'Initial value: {fun(phases):.2e} '
        f'Initial gradient norm: {optax.tree.norm(jax.grad(fun)(phases)):.2e}'
    )
    final_params, _ = run_opt(phases, fun, opt, max_iter=max_iter, tol=tol)
    print(
        f'Final value: {fun(final_params):.2e}, '
        f'Final gradient norm: {optax.tree.norm(jax.grad(fun)(final_params)):.2e}'
    )

    return final_params, losses

