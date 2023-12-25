import typing
import jax
import kernels
import jax.numpy as jnp
import jaxopt


class Dataset(typing.NamedTuple):
    xs: jax.Array
    ys: jax.Array


def pertubate_covariance_matrix(
    covariance_matrix: jax.Array, pertubation: jax.Array
) -> jax.Array:
    # Note: This function should only be used for with optimize.
    if covariance_matrix.ndim != 2:
        raise ValueError(
            f"covariance_matrix.ndim must be 2. covariance_matrix.ndim: {covariance_matrix.ndim}"
        )
    if pertubation.ndim != 0:
        raise ValueError(
            f"pertubation.ndim must be 0. pertubation.ndim: {pertubation.ndim}"
        )
    return covariance_matrix + (pertubation * jnp.eye(len(covariance_matrix)))


def get_negative_log_marginal_likelihood(
    covariance_matrix: jax.Array, ys: jax.Array
) -> jax.Array:
    if covariance_matrix.ndim != 2:
        raise ValueError(
            f"covariance_matrix.ndim must be 2. covariance_matrix.ndim: {covariance_matrix.ndim}"
        )
    if ys.ndim != 1:
        raise ValueError(f"ys.ndim must be 1. ys.ndim: {ys.ndim}")
    ys = jnp.expand_dims(ys, -1)
    l = jnp.linalg.cholesky(covariance_matrix)
    inv_L = jnp.linalg.inv(l)
    inv_L_T = jnp.linalg.inv(l.T)
    log_detK = jnp.sum(jnp.log(jnp.diag(l)))
    yT_invM_y = jnp.squeeze(ys.T @ inv_L_T @ inv_L @ ys)
    return (
        0.5 * yT_invM_y
        + 0.5 * 2 * log_detK
        + (len(covariance_matrix) / 2) * jnp.log(2 * jnp.pi)
    )


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
    kernel_dataset_xs_matmul_inverse_noisy_kernel_dataset_dataset = (
        jax.scipy.linalg.solve(
            noisy_kernel_dataset_dataset, kernel_dataset_xs, assume_a="pos"
        ).T
    )
    mean = kernel_dataset_xs_matmul_inverse_noisy_kernel_dataset_dataset @ dataset.ys
    kernel_xs_xs = kernel(state, xs, xs)
    covariance = kernel_xs_xs - (
        kernel_dataset_xs_matmul_inverse_noisy_kernel_dataset_dataset
        @ kernel_dataset_xs
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


def optimize(
    kernel: kernels.Kernel,
    initial_state: kernels.State,
    dataset: Dataset,
    max_iterations: int,
    tolerance: float,
    verbose: bool = False,
) -> tuple[kernels.State, jax.Array]:
    optimizer = jaxopt.LBFGS(
        lambda s: get_negative_log_marginal_likelihood(
            kernel(
                s,
                dataset.xs,
                dataset.xs,
            ),
            dataset.ys,
        ),
        maxiter=max_iterations,
        tol=tolerance,
        verbose=verbose,
    )
    opt_step = optimizer.run(initial_state)
    return opt_step.params, jnp.logical_and(
        jnp.logical_not(opt_step.state.failed_linesearch),
        jnp.isfinite(opt_step.state.error),
    )


