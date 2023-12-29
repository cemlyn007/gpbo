if __name__ == "__main__":
    import os
    from gpbo import gaussian_process, objective_functions, kernels, render
    import jax.numpy as jnp
    import jax.random
    import jax.experimental
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    import argparse
    import platform

    argument_parser = argparse.ArgumentParser("Gaussian Process Example")

    argument_parser.add_argument(
        "--plot_resolution",
        type=int,
        default=100,
        help="Number of points to plot along each axis",
    )
    argument_parser.add_argument(
        "--iterations",
        type=int,
        default=512,
        help="Number of iterations to run where points are sampled and the model is optimized",
    )
    argument_parser.add_argument(
        "--initial_dataset_size",
        type=int,
        default=1,
        help="Number of points to sample initially",
    )
    argument_parser.add_argument(
        "--optimize_max_iterations",
        type=int,
        default=500,
        help="Maximum number of iterations to run the optimizer",
    )
    argument_parser.add_argument(
        "--optimize_tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for the optimizer",
    )
    argument_parser.add_argument(
        "--optimize_penalty",
        type=float,
        default=10000,
        help="Penalty for the optimizer",
    )
    argument_parser.add_argument(
        "--use_auto_grad",
        action="store_true",
        help="Use automatic differentiation",
    )
    argument_parser.add_argument(
        "--use_x64", action="store_true", help="Use 64-bit floating point"
    )
    argument_parser.add_argument(
        "--plot_throughout", action="store_true", help="Plot throughout"
    )
    argument_parser.add_argument(
        "--objective_function",
        type=str,
        default="univariate",
        help="Objective function to use, options are univariate and six_hump_camel",
        choices=["univariate", "six_hump_camel"],
    )
    argument_parser.add_argument(
        "--noisy_objective_function",
        type=float,
        default=0.0,
        help="Objective function noise",
    )

    arguments = argument_parser.parse_args()

    if platform.system() == "Darwin":
        SAVE_PATH = os.path.join(os.getcwd(), "renders.tmp")
    else:
        SAVE_PATH = os.path.join(os.getcwd(), "renders")

    log_amplitude = math.log(1.0)
    log_length_scale = math.log(0.5)
    log_noise_scale = math.log(0.5 * math.exp(log_amplitude))

    figure = plt.figure(tight_layout=True, figsize=(12, 4))

    with jax.experimental.enable_x64(arguments.use_x64):
        BOUNDS = (
            kernels.State(
                jnp.array(-3, dtype=float),
                jnp.array(-3, dtype=float),
                jnp.array(-3, dtype=float),
            ),
            kernels.State(
                jnp.array(3, dtype=float),
                jnp.array(3, dtype=float),
                jnp.array(3, dtype=float),
            ),
        )

        print(jnp.array(1, float).dtype)
        if arguments.objective_function == "univariate":
            objective_function = objective_functions.UnivariateObjectiveFunction()
        elif arguments.objective_function == "six_hump_camel":
            objective_function = objective_functions.SixHumpCamelObjectiveFunction()
        else:
            raise ValueError(f"Unknown objective function: {arguments.objective_function}")

        if arguments.noisy_objective_function > 0.0:
            objective_function = objective_functions.NoisyObjectiveFunction(
                objective_function, arguments.noisy_objective_function
            )

        objective_function = objective_functions.JitObjectiveFunction(
            objective_function
        )

        kernel = kernels.gaussian

        key = jax.random.PRNGKey(0)

        dataset = None

        state = kernels.State(
            jnp.asarray(log_amplitude),
            jnp.asarray(log_length_scale),
            jnp.asarray(log_noise_scale),
        )

        optimize = jax.jit(gaussian_process.optimize, static_argnums=(0, 3, 4, 6, 7, 8))

        get_mean_and_std = jax.jit(
            gaussian_process.get_mean_and_std, static_argnums=(0,)
        )

        mesh_grid = objective_functions.get_mesh_grid(
            [(boundary, arguments.plot_resolution) for boundary in objective_function.dataset_bounds],
            False,
        )
        ticks = tuple(
            np.asarray(objective_functions.get_ticks(boundary, arguments.plot_resolution))
            for boundary in objective_function.dataset_bounds
        )
        grid_xs = jnp.dstack(mesh_grid).reshape(-1, len(mesh_grid))
        grid_ys = np.asarray(
            objective_function.evaluate(jax.random.PRNGKey(0), *mesh_grid)
        )

        negative_log_marginal_likelihoods_xs = []
        negative_log_marginal_likelihoods = []

        key, sample_key, evaluate_key = jax.random.split(key, 3)
        xs = objective_functions.sample(
            sample_key,
            arguments.initial_dataset_size,
            objective_function.dataset_bounds,
        )

        xs_args = tuple(xs[:, i] for i in range(xs.shape[1])) if xs.ndim > 1 else (xs,)
        ys = objective_function.evaluate(evaluate_key, *xs_args)

        dataset = gaussian_process.Dataset(xs, ys)

        for i in range(arguments.iterations):
            key, sample_key, evaluate_key = jax.random.split(key, 3)
            xs = objective_functions.sample(
                sample_key,
                1,
                objective_function.dataset_bounds,
            )

            xs_args = (
                tuple(xs[:, j] for j in range(xs.shape[1])) if xs.ndim > 1 else (xs,)
            )
            ys = objective_function.evaluate(evaluate_key, *xs_args)

            dataset = dataset._replace(
                xs=jnp.concatenate([dataset.xs, xs], axis=0),
                ys=jnp.concatenate([dataset.ys, ys], axis=0),
            )

            state, ok = optimize(
                kernel,
                state,
                dataset,
                arguments.optimize_max_iterations,
                arguments.optimize_tolerance,
                BOUNDS,
                arguments.optimize_penalty,
                use_auto_grad=arguments.use_auto_grad,
                verbose=False,
            )
            print(i, state)
            assert ok.item(), f"Optimization failed on iteration {i}"

            if arguments.plot_throughout or i == (arguments.iterations - 1):
                negative_log_marginal_likelihoods_xs.append(i)
                negative_log_marginal_likelihoods.append(
                    -gaussian_process.get_log_marginal_likelihood(
                        kernel(state, dataset.xs, dataset.xs), dataset.ys
                    )
                )

                mean, std = get_mean_and_std(kernel, state, dataset, grid_xs)

                mean = np.asarray(
                    mean.reshape((arguments.plot_resolution,) * len(objective_function.dataset_bounds))
                )

                std = np.asarray(
                    std.reshape((arguments.plot_resolution,) * len(objective_function.dataset_bounds))
                )
                std = np.where(np.isfinite(std), std, -1.0)

                figure.clear()
                render.plot(
                    ticks,
                    grid_ys,
                    mean,
                    std,
                    np.asarray(dataset.xs),
                    np.asarray(dataset.ys),
                    figure,
                )
                if not os.path.exists(SAVE_PATH):
                    os.makedirs(SAVE_PATH, exist_ok=True)
                figure.savefig(os.path.join(SAVE_PATH, f"{i}.png"))

    plt.close(figure)

    plt.plot(negative_log_marginal_likelihoods_xs, negative_log_marginal_likelihoods)
    plt.savefig(os.path.join(SAVE_PATH, "negative_log_marginal_likelihoods.png"))
