if __name__ == "__main__":
    import os
    from gpbo import (
        gaussian_process,
        objective_functions,
        kernels,
        render,
        datasets,
        transformers,
        io,
    )
    import jax.numpy as jnp
    import jax.random
    import jax.experimental
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    import argparse
    import tqdm
    import platform
    import jaxlib.xla_extension

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
        "--cast_x64_to_x32_objective_function",
        action="store_true",
        help="Cast 32-bit floating point for the objective function",
    )
    argument_parser.add_argument(
        "--plot_throughout", action="store_true", help="Plot throughout"
    )
    argument_parser.add_argument(
        "--objective_function",
        type=str,
        default="univariate",
        help="Objective function to use, options are univariate, six_hump_camel, mnist_1d, mnist_2d and npy",
        choices=["univariate", "six_hump_camel", "mnist_1d", "mnist_2d", "npy"],
    )
    argument_parser.add_argument(
        "--noisy_objective_function",
        type=float,
        default=0.0,
        help="Objective function noise",
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        default=None,
        help="Transform to use for the dataset, options are standardize, min_max_scale, standardize_xs_only, mean_center_xs_only and mean_center",
        choices=[
            "standardize",
            "min_max_scale",
            "mean_center",
            "standardize_xs_only",
            "mean_center_xs_only",
            "none",
            None,
        ],
    )
    argument_parser.add_argument(
        "--kernel",
        type=str,
        default=None,
        help="Kernel to use for the Gaussian Process, options are gaussian and matern",
        choices=[
            "gaussian",
            "matern",
        ],
    )
    argument_parser.add_argument(
        "--grid_xs_npy_path",
        type=str,
        help="Path to the grid xs npy file if using the npy objective function",
        default=None,
    )
    argument_parser.add_argument(
        "--grid_ys_npy_path",
        type=str,
        help="Path to the grid ys npy file if using the npy objective function",
        default=None,
    )
    argument_parser.add_argument(
        "--save_path",
        type=str,
        default=None,
    )

    arguments = argument_parser.parse_args()

    if arguments.kernel is None:
        raise ValueError("Kernel flag must be specified")

    if arguments.save_path is None:
        save_path = (
            os.path.join(os.getcwd(), "results.tmp")
            if platform.system() == "Darwin"
            else os.path.join(os.getcwd(), "results")
        )
        save_path = os.path.join(save_path, arguments.objective_function, arguments.kernel, arguments.transform)
    else:
        save_path = arguments.save_path

    os.makedirs(save_path, exist_ok=True)

    log_amplitude = math.log(1.0)
    log_length_scale = math.log(0.5)
    log_noise_scale = math.log(0.5 * math.exp(log_amplitude))

    figure = plt.figure(tight_layout=True, figsize=(12, 4))

    with jax.experimental.enable_x64(arguments.use_x64):
        if arguments.kernel == "gaussian":
            kernel = kernels.gaussian
        elif arguments.kernel == "matern":
            kernel = kernels.matern
        else:
            raise ValueError(f"Unknown kernel: {arguments.kernel}")

        LOWER_BOUND = -10.0
        UPPER_BOUND = 10.0
        BOUNDS = (
            kernels.State(
                jnp.array(LOWER_BOUND, dtype=float),
                jnp.array(LOWER_BOUND, dtype=float),
                jnp.array(LOWER_BOUND, dtype=float),
            ),
            kernels.State(
                jnp.array(UPPER_BOUND, dtype=float),
                jnp.array(UPPER_BOUND, dtype=float),
                jnp.array(UPPER_BOUND, dtype=float),
            ),
        )
        BOUNDS = None

        transformer: transformers.Transformer = {
            "standardize": transformers.Standardizer,
            "min_max_scale": transformers.MinMaxScaler,
            "mean_center": transformers.MeanCenterer,
            "standardize_xs_only": transformers.StandardizerXsOnly,
            "mean_center_xs_only": transformers.MeanCentererXsOnly,
            "none": transformers.Identity,
            None: transformers.Identity,
        }[arguments.transform]()

        print(jnp.array(1, float).dtype)
        if arguments.objective_function == "npy":
            grid_xs = jnp.load(arguments.grid_xs_npy_path)
            grid_ys = jnp.load(arguments.grid_ys_npy_path)
            jnp.save(os.path.join(save_path, "grid_xs.npy"), grid_xs)
            jnp.save(os.path.join(save_path, "grid_ys.npy"), grid_ys)
            if grid_xs.shape[0] == 1:
                tmp = jnp.dstack(jnp.meshgrid(*grid_xs)).flatten()
            else:
                tmp = jnp.dstack(jnp.meshgrid(*grid_xs)).reshape(-1, grid_xs.shape[0])
            objective_function = objective_functions.MeshGridObjectiveFunction(tmp, grid_ys.flatten())
        else:
            if arguments.objective_function == "univariate":
                objective_function = objective_functions.UnivariateObjectiveFunction()
            elif arguments.objective_function == "six_hump_camel":
                objective_function = objective_functions.SixHumpCamelObjectiveFunction()
            elif arguments.objective_function == "mnist_1d":
                objective_function = objective_functions.MnistObjectiveFunction(
                    "/tmp/mnist", False, 100, jax.devices()[0]
                )
            elif arguments.objective_function == "mnist_2d":
                objective_function = objective_functions.MnistObjectiveFunction(
                    "/tmp/mnist", True, 100, jax.devices()[0]
                )
            else:
                raise ValueError(
                    f"Unknown objective function: {arguments.objective_function}"
                )

            if arguments.noisy_objective_function > 0.0:
                objective_function = objective_functions.NoisyObjectiveFunction(
                    objective_function, arguments.noisy_objective_function
                )

            if arguments.cast_x64_to_x32_objective_function:
                objective_function = objective_functions.DtypeCasterObjectiveFunction(
                    objective_function
                )

            objective_function = objective_functions.JitObjectiveFunction(
                objective_function
            )

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

        if isinstance(objective_function, objective_functions.MeshGridObjectiveFunction):
            ticks = tuple(np.array(grid_xs[i])
                          for i in range(grid_xs.shape[0]))
            
            if grid_xs.shape[0] == 1:
                grid_xs = jnp.dstack(jnp.meshgrid(*grid_xs)).flatten()
            else:
                grid_xs = jnp.dstack(jnp.meshgrid(*grid_xs)).reshape(-1, grid_xs.shape[0])

            unsampled_indices = list(range(grid_xs.shape[0]))
            def sample_index(key):
                index = jax.random.choice(key, jnp.array(unsampled_indices), (), replace=False)
                unsampled_indices.remove(index.item())
                return index.item()
            
            keys = jax.random.split(key, arguments.initial_dataset_size)
            indices = jnp.array([sample_index(key) for key in keys])

            xs = grid_xs[indices]
            ys = grid_ys.flatten()[indices]
        else:
            mesh_grid = objective_functions.utils.get_mesh_grid(
                [
                    (boundary, arguments.plot_resolution)
                    for boundary in objective_function.dataset_bounds
                ],
                False,
            )
            ticks = tuple(
                np.asarray(
                    objective_functions.utils.get_ticks(boundary, arguments.plot_resolution)
                )
                for boundary in objective_function.dataset_bounds
            )
            grid_xs = jnp.dstack(mesh_grid).reshape(-1, len(mesh_grid))

            try:
                grid_ys = np.asarray(
                    objective_function.evaluate(jax.random.PRNGKey(0), *mesh_grid)
                )
            except jaxlib.xla_extension.XlaRuntimeError:
                grid_ys = np.asarray(
                    [
                        objective_function.evaluate(
                            jax.random.PRNGKey(0),
                            *(
                                grid_xs[ii : ii + 1, iii]
                                for iii in range(len(objective_function.dataset_bounds))
                            ),
                        )
                        for ii in tqdm.tqdm(
                            range(grid_xs.shape[0]),
                            total=grid_xs.shape[0],
                            desc="Loading grid objective function values...",
                        )
                    ]
                )
                grid_ys = grid_ys.reshape(
                    (arguments.plot_resolution,) * len(objective_function.dataset_bounds)
                )

            jnp.save(os.path.join(save_path, "grid_xs.npy"), ticks)
            jnp.save(os.path.join(save_path, "grid_ys.npy"), grid_ys)

            key, sample_key, evaluate_key = jax.random.split(key, 3)
            xs = objective_functions.utils.sample(
                sample_key,
                arguments.initial_dataset_size,
                objective_function.dataset_bounds,
            )

            xs_args = tuple(xs[:, i] for i in range(xs.shape[1])) if xs.ndim > 1 else (xs,)
            try:
                ys = objective_function.evaluate(evaluate_key, *xs_args)
            except jaxlib.xla_extension.XlaRuntimeError:
                ys = jnp.concatenate(
                    [
                        objective_function.evaluate(
                            evaluate_key, *(arg[ii : ii + 1] for arg in xs_args)
                        )
                        for ii in range(arguments.initial_dataset_size)
                    ]
                )

        

        negative_log_marginal_likelihoods_xs = []
        negative_log_marginal_likelihoods = []

        dataset = datasets.Dataset(xs, ys)

        cpu_device = jax.devices("cpu")[0]

        util_device = jax.devices()[0]

        for i in range(arguments.iterations):
            key, sample_key, evaluate_key = jax.random.split(key, 3)
            if isinstance(objective_function, objective_functions.MeshGridObjectiveFunction):
                xs = grid_xs[sample_index(sample_key)]
                if len(xs.shape) == 0:
                    xs = jnp.expand_dims(xs, 0)
            else:
                xs = objective_functions.utils.sample(
                    sample_key,
                    1,
                    objective_function.dataset_bounds,
                )
                xs = jnp.reshape(xs, (len(objective_function.dataset_bounds,)))

            assert xs.shape == (len(objective_function.dataset_bounds),)

            if len(objective_function.dataset_bounds) == 1:
                xs_args = (xs,)
            else:
                xs_args = tuple(xs[j] for j in range(xs.shape[0]))

            ys = objective_function.evaluate(evaluate_key, *xs_args)

            assert xs.shape == (len(objective_function.dataset_bounds),)

            if len(objective_function.dataset_bounds) > 1:
                xs = jnp.reshape(xs, (1, len(objective_function.dataset_bounds)))

            ys = jnp.reshape(ys, (1,))

            dataset = dataset._replace(
                xs=jnp.concatenate([dataset.xs, xs], axis=0),
                ys=jnp.concatenate([dataset.ys, ys], axis=0),
            )
            (
                transformed_dataset,
                dataset_center,
                dataset_scale,
            ) = transformer.transform_dataset(dataset)

            state, ok = optimize(
                kernel,
                state,
                transformed_dataset,
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
                        kernel,
                        state,
                        transformed_dataset,
                    )
                )

                try:
                    transformed_mean, transformed_std = get_mean_and_std(
                        kernel,
                        jax.device_put(state, util_device),
                        jax.device_put(transformed_dataset, util_device),
                        transformer.transform_values(
                            jax.device_put(grid_xs, util_device),
                            None if dataset_center is None else dataset_center.xs,
                            None if dataset_scale is None else dataset_scale.xs,
                        ),
                    )
                except jaxlib.xla_extension.XlaRuntimeError:
                    util_device = cpu_device
                    transformed_mean, transformed_std = get_mean_and_std(
                        kernel,
                        jax.device_put(state, cpu_device),
                        jax.device_put(transformed_dataset, cpu_device),
                        transformer.transform_values(
                            jax.device_put(grid_xs, cpu_device),
                            None if dataset_center is None else dataset_center.xs,
                            None if dataset_scale is None else dataset_scale.xs,
                        ),
                    )

                mean = transformer.inverse_transform_values(
                    transformed_mean,
                    None if dataset_center is None else dataset_center.ys,
                    None if dataset_scale is None else dataset_scale.ys,
                )
                std = transformer.inverse_transform_y_stds(
                    transformed_std,
                    None if dataset_center is None else dataset_center.ys,
                    None if dataset_scale is None else dataset_scale.ys,
                )

                mean = np.asarray(
                    mean.reshape(
                        (arguments.plot_resolution,)
                        * len(objective_function.dataset_bounds)
                    )
                )

                std = np.asarray(
                    std.reshape(
                        (arguments.plot_resolution,)
                        * len(objective_function.dataset_bounds)
                    )
                )
                std = np.where(np.isfinite(std), std, -1.0)

                figure.clear()
                render.plot(
                    ticks,
                    grid_ys,
                    mean,
                    std,
                    None,
                    np.asarray(dataset.xs),
                    np.asarray(dataset.ys),
                    figure,
                )
                step_save_path = os.path.join(save_path, str(i))
                if not os.path.exists(step_save_path):
                    os.makedirs(step_save_path, exist_ok=True)
                
                figure.savefig(os.path.join(step_save_path, "figure.png"))
                io.write_csv(
                    dataset,
                    os.path.join(step_save_path, "dataset.csv"),
                )
                np.save(os.path.join(step_save_path, "mean.npy"), mean)
                np.save(os.path.join(step_save_path, "std.npy"), std)

    plt.close(figure)

    plt.plot(negative_log_marginal_likelihoods_xs, negative_log_marginal_likelihoods)
    plt.savefig(os.path.join(save_path, "negative_log_marginal_likelihoods.png"))
