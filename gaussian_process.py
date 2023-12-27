import typing
import jax
import kernels
import jax.numpy as jnp
import jaxopt

# Covariance matrices are symmetric and positive semi-definite, so we can use the "pos" option.
ASSUME_A = "pos"


class Dataset(typing.NamedTuple):
    xs: jax.Array
    ys: jax.Array


def get_log_marginal_likelihood(covariance_matrix: jax.Array, ys: jax.Array):
    if covariance_matrix.ndim != 2:
        raise ValueError(
            f"covariance_matrix.ndim must be 2. covariance_matrix.ndim: {covariance_matrix.ndim}"
        )
    if ys.ndim != 1:
        raise ValueError(f"ys.ndim must be 1. ys.ndim: {ys.ndim}")

    sign, slogdet = jnp.linalg.slogdet(covariance_matrix)
    return (
        -0.5 * ys.T @ jax.scipy.linalg.solve(covariance_matrix, ys, assume_a=ASSUME_A)
        - 0.5 * sign * slogdet
    )


def get_gradient_log_marginal_likelihood(
    kernel: kernels.Kernel,
    state: kernels.State,
    dataset: Dataset,
) -> kernels.State:
    K = kernel(state, dataset.xs, dataset.xs)
    noise = kernels.noise_scale_squared(state) * jnp.identity(K.shape[0])
    K_noise = K + noise
    K_noise_inv = jnp.linalg.inv(K_noise)
    y = dataset.ys

    alpha = jax.scipy.linalg.solve(K_noise, y, assume_a=ASSUME_A)

    length_scale = kernels.length_scale(state)

    squared_distances = kernels.euclidean_squared_distance_matrix(
        dataset.xs, dataset.xs
    )

    if kernel is kernels.gaussian:
        grad_log_amplitude = jnp.trace((alpha @ alpha.T - K_noise_inv) @ K)
        grad_log_length_scale = jnp.divide(
            jnp.trace((alpha @ alpha.T - K_noise_inv) @ (squared_distances * K)),
            jnp.square(length_scale),
        )
        grad_log_noise_scale = jnp.trace((alpha @ alpha.T - K_noise_inv) @ noise)
        gradient = kernels.State(
            grad_log_amplitude, grad_log_length_scale, grad_log_noise_scale
        )
    else:
        raise NotImplementedError
    return gradient


def get_mean_and_covariance(
    kernel: kernels.Kernel,
    state: kernels.State,
    dataset: Dataset,
    xs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    noisy_kernel_dataset_dataset = kernel(state, dataset.xs, dataset.xs) + (
        kernels.noise_scale_squared(state) * jnp.eye(dataset.xs.shape[0])
    )
    kernel_dataset_xs = kernel(state, dataset.xs, xs)
    solve_kwargs = dict(
        assume_a=ASSUME_A,
    )
    mean = kernel_dataset_xs.T @ jax.scipy.linalg.solve(
        noisy_kernel_dataset_dataset, dataset.ys, **solve_kwargs
    )
    covariance = kernel(state, xs, xs) - kernel_dataset_xs.T @ jax.scipy.linalg.solve(
        noisy_kernel_dataset_dataset, kernel_dataset_xs, **solve_kwargs
    )
    return mean, covariance


def get_mean_and_variance(
    kernel: kernels.Kernel,
    state: kernels.State,
    dataset: Dataset,
    xs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    mean, covariance = get_mean_and_covariance(kernel, state, dataset, xs)
    variance = jnp.diag(covariance)
    return mean, variance


def get_mean_and_std(
    kernel: kernels.Kernel,
    state: kernels.State,
    dataset: Dataset,
    xs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    mean, variance = get_mean_and_variance(kernel, state, dataset, xs)
    std = jnp.sqrt(variance)
    return mean, std


def get_log_predictive_density(
    kernel: kernels.Kernel,
    state: kernels.State,
    dataset: Dataset,
    xs: jax.Array,
    ys: jax.Array,
) -> jax.Array:
    mean, variance = get_mean_and_variance(kernel, state, dataset, xs)
    std = jnp.sqrt(variance + kernels.noise_scale_squared(state))
    return jax.scipy.stats.norm.logpdf(ys, loc=mean, scale=std)


def sample(
    kernel: kernels.Kernel,
    state: kernels.State,
    dataset: Dataset,
    key: jax.Array,
    xs: jax.Array,
) -> jax.Array:
    mean, covariance = get_mean_and_covariance(kernel, state, dataset, xs)
    return jax.random.multivariate_normal(key, mean, covariance)


def get_signal_noise_ratio_loss(
    state: kernels.State, max_snr_ratio: float
) -> jax.Array:
    """Adapted from https://infallible-thompson-49de36.netlify.app/#section-6.2.2"""
    ratio = jnp.divide(jnp.exp(state.log_amplitude), jnp.exp(state.log_noise_scale))
    snr_loss = jnp.power(jnp.divide(jnp.log(ratio), jnp.log(max_snr_ratio)), 50)
    return snr_loss


def get_gradient_signal_noise_ratio_loss(
    state: kernels.State, max_snr_ratio: float
) -> kernels.State:
    """Adapted from https://infallible-thompson-49de36.netlify.app/#section-6.2.2"""
    gradient_log_amplitude = (
        jnp.divide(
            50, jnp.multiply(jnp.log(max_snr_ratio), jnp.exp(state.log_amplitude))
        )
        * jnp.power(
            jnp.divide(
                jnp.log(
                    jnp.divide(
                        jnp.exp(state.log_amplitude), jnp.exp(state.log_noise_scale)
                    )
                ),
                jnp.log(max_snr_ratio),
            ),
            49,
        )
        * jnp.exp(state.log_amplitude)
    )
    gradient_log_noise_scale = (
        jnp.divide(
            -50, jnp.multiply(jnp.log(max_snr_ratio), jnp.exp(state.log_noise_scale))
        )
        * jnp.power(
            jnp.divide(
                jnp.log(
                    jnp.divide(
                        jnp.exp(state.log_amplitude), jnp.exp(state.log_noise_scale)
                    )
                ),
                jnp.log(max_snr_ratio),
            ),
            49,
        )
        * jnp.exp(state.log_noise_scale)
    )
    gradient = kernels.State(
        gradient_log_amplitude,
        jnp.zeros_like(state.log_length_scale),
        gradient_log_noise_scale,
    )
    return gradient


def optimize(
    kernel: kernels.Kernel,
    initial_state: kernels.State,
    dataset: Dataset,
    max_iterations: int,
    tolerance: float,
    bounds: tuple[kernels.State, kernels.State],
    max_snr_ratio: float,
    use_auto_grad: bool = False,
    verbose: bool = False,
) -> tuple[kernels.State, jax.Array]:
    def value(s: kernels.State) -> jax.Array:
        negative_log_marginal_likelihood = -get_log_marginal_likelihood(
            kernel(
                s,
                dataset.xs,
                dataset.xs,
            ),
            dataset.ys,
        )
        if max_snr_ratio > 0.0:
            snr_loss = get_signal_noise_ratio_loss(s, max_snr_ratio)
            negative_log_marginal_likelihood += snr_loss

        if verbose:
            if max_snr_ratio > 0.0:
                jax.debug.print(
                    "state={state}\nsnr={snr}",
                    state=s,
                    snr=snr_loss,
                )
            else:
                jax.debug.print(
                    "state={state}",
                    state=s,
                )
        return negative_log_marginal_likelihood

    def value_and_grad(s: kernels.State) -> tuple[jax.Array, kernels.State]:
        negative_log_marginal_likelihood = -get_log_marginal_likelihood(
            kernel(
                s,
                dataset.xs,
                dataset.xs,
            ),
            dataset.ys,
        )
        gradient_negative_log_marginal_likelihood = jax.tree_map(
            lambda x: -x, get_gradient_log_marginal_likelihood(kernel, s, dataset)
        )

        loss = negative_log_marginal_likelihood
        gradient_loss = gradient_negative_log_marginal_likelihood
        if max_snr_ratio > 0.0:
            # https://infallible-thompson-49de36.netlify.app/#section-6.2.2
            snr_loss = get_signal_noise_ratio_loss(s, max_snr_ratio)
            gradient_snr_loss = get_gradient_signal_noise_ratio_loss(s, max_snr_ratio)
            loss += snr_loss
            gradient_loss = jax.tree_map(
                jnp.add,
                gradient_loss,
                gradient_snr_loss,
            )

        if verbose:
            if max_snr_ratio > 0.0:
                jax.debug.print(
                    "state={state}\nsnr={snr}\ngradient_snr={gradient_snr}\ngradient={gradient}",
                    state=s,
                    snr=snr_loss,
                    gradient_snr=gradient_snr_loss,
                    gradient=gradient_negative_log_marginal_likelihood,
                )
            else:
                jax.debug.print(
                    "state={state}\ngradient={gradient}",
                    state=s,
                    gradient=gradient_negative_log_marginal_likelihood,
                )

        return loss, gradient_loss

    kwargs = dict(
        maxiter=max_iterations,
        tol=tolerance,
        verbose=verbose,
        history_size=max_iterations,
    )

    if use_auto_grad:
        optimizer = jaxopt.LBFGSB(value, **kwargs)
    else:
        optimizer = jaxopt.LBFGSB(value_and_grad, value_and_grad=True, **kwargs)
    opt_step = optimizer.run(
        initial_state,
        bounds,
    )
    ok = jnp.isfinite(opt_step.state.error)
    return (opt_step.params, ok)
