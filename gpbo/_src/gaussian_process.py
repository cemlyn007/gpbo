import jax
import jax.numpy as jnp
import jaxopt

from gpbo._src import datasets, kernels


def get_log_marginal_likelihood(
    kernel: kernels.Kernel,
    state: kernels.State,
    dataset: datasets.Dataset,
) -> jax.Array:
    covariance_matrix = kernel(
        state,
        dataset.xs,
        dataset.xs,
    ) + kernels.noise_scale_squared(state) * jnp.identity(dataset.xs.shape[0])
    if covariance_matrix.ndim != 2:
        raise ValueError(
            f"covariance_matrix.ndim must be 2. covariance_matrix.ndim: {covariance_matrix.ndim}"
        )
    if dataset.ys.ndim != 1:
        raise ValueError(f"ys.ndim must be 1. ys.ndim: {dataset.ys.ndim}")
    # Challenges for Gaussian processes - Imperial - Slide 5, Equation 7:
    L, lower = jax.scipy.linalg.cho_factor(covariance_matrix, lower=True)
    x = jax.scipy.linalg.solve_triangular(L, dataset.ys, lower=lower)
    part_a = (
        -0.5
        * dataset.ys.T
        @ jax.scipy.linalg.solve_triangular(L, x, lower=lower, trans="T")
    )
    part_b = -jnp.sum(jnp.log(jnp.diagonal(L)))
    return part_a + part_b


def get_gradient_log_marginal_likelihood(
    kernel: kernels.Kernel,
    state: kernels.State,
    dataset: datasets.Dataset,
) -> kernels.State:
    covariance_matrix = kernel(state, dataset.xs, dataset.xs)
    identity = jnp.identity(covariance_matrix.shape[0])
    noise = kernels.noise_scale_squared(state) * identity
    noised_covariance_matrix = covariance_matrix + noise

    L = jax.scipy.linalg.cho_factor(noised_covariance_matrix, lower=True)
    alpha = jax.scipy.linalg.cho_solve(L, dataset.ys)
    alpha = jnp.expand_dims(alpha, axis=-1)

    length_scale = kernels.length_scale(state)

    squared_distances = kernels.euclidean_squared_distance_matrix(
        dataset.xs, dataset.xs
    )

    if kernel is kernels.gaussian:
        dkernel_dstate = kernels.State(
            2 * covariance_matrix,
            jnp.divide(
                covariance_matrix * squared_distances,
                jnp.square(length_scale),
            ),
            2 * noise,
        )
    elif kernel is kernels.matern:
        tmp = jnp.divide(jnp.sqrt(3 * squared_distances), length_scale)
        dkernel_dstate = kernels.State(
            2 * covariance_matrix,
            kernels.amplitude_squared(state) * jnp.square(tmp) * jnp.exp(-tmp),
            2 * noise,
        )
    else:
        # TODO: Kernels should implement their own methods instead of being hardcoded here.
        raise NotImplementedError(f"kernel: {kernel} gradient not implemented")

    gradient = jax.tree_map(
        lambda dkernel_dtheta: (
            0.5
            * jnp.trace(
                (
                    alpha @ alpha.T @ dkernel_dtheta
                    - jax.scipy.linalg.cho_solve(L, dkernel_dtheta)
                )
            )
        ),
        dkernel_dstate,
    )
    return gradient


def get_mean_and_covariance(
    kernel: kernels.Kernel,
    state: kernels.State,
    dataset: datasets.Dataset,
    xs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    identity = jnp.identity(dataset.xs.shape[0])
    noisy_kernel_dataset_dataset = kernel(state, dataset.xs, dataset.xs) + (
        kernels.noise_scale_squared(state) * identity
    )
    kernel_dataset_xs = kernel(state, dataset.xs, xs)

    L = jax.scipy.linalg.cho_factor(noisy_kernel_dataset_dataset, lower=True)
    mean = kernel_dataset_xs.T @ jax.scipy.linalg.cho_solve(L, dataset.ys)

    covariance = kernel(
        state, xs, xs
    ) - kernel_dataset_xs.T @ jax.scipy.linalg.cho_solve(L, kernel_dataset_xs)
    return mean, covariance


def get_mean_and_variance(
    kernel: kernels.Kernel,
    state: kernels.State,
    dataset: datasets.Dataset,
    xs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    mean, covariance = get_mean_and_covariance(kernel, state, dataset, xs)
    variance = jnp.diag(covariance)
    return mean, variance


def get_mean_and_std(
    kernel: kernels.Kernel,
    state: kernels.State,
    dataset: datasets.Dataset,
    xs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    mean, variance = get_mean_and_variance(kernel, state, dataset, xs)
    std = jnp.sqrt(variance)
    return mean, std


def get_log_predictive_density(
    kernel: kernels.Kernel,
    state: kernels.State,
    dataset: datasets.Dataset,
    xs: jax.Array,
    ys: jax.Array,
) -> jax.Array:
    mean, variance = get_mean_and_variance(kernel, state, dataset, xs)
    std = jnp.sqrt(variance + kernels.noise_scale_squared(state))
    return jax.scipy.stats.norm.logpdf(ys, loc=mean, scale=std)


def sample(
    kernel: kernels.Kernel,
    state: kernels.State,
    dataset: datasets.Dataset,
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
    dataset: datasets.Dataset,
    max_iterations: int,
    tolerance: float,
    bounds: tuple[kernels.State, kernels.State],
    max_snr_ratio: float,
    use_auto_grad: bool = False,
    verbose: bool = False,
) -> tuple[kernels.State, jax.Array]:
    def value(s: kernels.State) -> jax.Array:
        negative_log_marginal_likelihood = -get_log_marginal_likelihood(
            kernel, s, dataset
        )
        if max_snr_ratio > 0.0:
            snr_loss = get_signal_noise_ratio_loss(s, max_snr_ratio)
            negative_log_marginal_likelihood += snr_loss

        if verbose:
            if max_snr_ratio > 0.0:
                jax.debug.print(
                    "state={state}\n"
                    "negative_log_marginal_likelihood={negative_log_marginal_likelihood}\n"
                    "snr={snr}\n",
                    state=s,
                    negative_log_marginal_likelihood=negative_log_marginal_likelihood,
                    snr=snr_loss,
                )
            else:
                jax.debug.print(
                    "state={state}\n"
                    "negative_log_marginal_likelihood={negative_log_marginal_likelihood}\n",
                    state=s,
                    negative_log_marginal_likelihood=negative_log_marginal_likelihood,
                )
        return negative_log_marginal_likelihood

    def value_and_grad(s: kernels.State) -> tuple[jax.Array, kernels.State]:
        negative_log_marginal_likelihood = -get_log_marginal_likelihood(
            kernel,
            s,
            dataset,
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
                    "state={state}\n"
                    "negative_log_marginal_likelihood={negative_log_marginal_likelihood}\n"
                    "gradient_negative_log_marginal_likelihood={gradient_negative_log_marginal_likelihood}\n"
                    "snr={snr}\n"
                    "gradient_snr={gradient_snr}\n",
                    state=s,
                    negative_log_marginal_likelihood=negative_log_marginal_likelihood,
                    gradient_negative_log_marginal_likelihood=gradient_negative_log_marginal_likelihood,
                    snr=snr_loss,
                    gradient_snr=gradient_snr_loss,
                )
            else:
                jax.debug.print(
                    "state={state}\n"
                    "negative_log_marginal_likelihood={negative_log_marginal_likelihood}\n"
                    "gradient_negative_log_marginal_likelihood={gradient_negative_log_marginal_likelihood}\n",
                    state=s,
                    negative_log_marginal_likelihood=negative_log_marginal_likelihood,
                    gradient_negative_log_marginal_likelihood=gradient_negative_log_marginal_likelihood,
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


def multi_optimize(
    kernel: kernels.Kernel,
    initial_states: tuple[kernels.State, ...],
    dataset: datasets.Dataset,
    max_iterations: int,
    tolerance: float,
    bounds: tuple[kernels.State, kernels.State],
    max_snr_ratio: float,
    use_auto_grad: bool = False,
    verbose: bool = False,
) -> tuple[kernels.State, jax.Array]:
    total_states = len(initial_states)
    initial_state = initial_states[0]
    initial_states = jax.tree_map(lambda *xs: jnp.stack(xs), *initial_states)

    def body_fun(i, val):
        (best_state, best_log_marginal_likelihood) = val
        assert isinstance(best_state, kernels.State)
        assert isinstance(best_log_marginal_likelihood, jax.Array)
        state, ok = optimize(
            kernel,
            jax.tree_map(lambda x: x[i], initial_states),
            dataset,
            max_iterations,
            tolerance,
            bounds,
            max_snr_ratio,
            use_auto_grad,
            verbose,
        )
        log_marginal_likelihood = get_log_marginal_likelihood(kernel, state, dataset)
        if verbose:
            jax.debug.print(
                "i={i}, log_marginal_likelihood={log_marginal_likelihood}\n",
                i=i,
                log_marginal_likelihood=log_marginal_likelihood,
            )
        better = log_marginal_likelihood > best_log_marginal_likelihood
        ok_and_better = jnp.logical_and(ok, better)
        best_state = jax.tree_map(
            lambda x, y: jnp.where(ok_and_better, x, y), state, best_state
        )
        best_log_marginal_likelihood = jnp.where(
            ok_and_better, log_marginal_likelihood, best_log_marginal_likelihood
        )
        assert isinstance(best_state, kernels.State)
        assert isinstance(best_log_marginal_likelihood, jax.Array)
        return (best_state, best_log_marginal_likelihood)

    return jax.lax.fori_loop(0, total_states, body_fun, (initial_state, -jnp.inf))
