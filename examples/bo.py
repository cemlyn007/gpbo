if __name__ == "__main__":
    from gpbo import (
        datasets,
        gaussian_process,
        objective_functions,
        kernels,
        render,
        acquisition_functions,
        transformers,
        io
    )

    import math

    import numpy as np

    import platform
    import os
    import jaxlib.xla_extension
    import tqdm

    import jax
    import jax.numpy as jnp
    import argparse
    import jax.experimental
    import matplotlib.pyplot as plt

    argument_parser = argparse.ArgumentParser(
        "Bayesian Optimization using a Gaussian Process Example"
    )

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
        "--objective_function",
        type=str,
        default="univariate",
        help="Objective function to use, options are univariate and six_hump_camel",
        choices=["univariate", "six_hump_camel", "mnist_1d", "mnist_2d", "npy"],
    )
    argument_parser.add_argument(
        "--noisy_objective_function",
        type=float,
        default=0.0,
        help="Objective function noise",
    )
    argument_parser.add_argument(
        "--acquisition_function",
        type=str,
        default="expected_improvement",
        help="Acquisition function to use, options are expected_improvement and lower_confidence_bound",
        choices=["expected_improvement", "lower_confidence_bound"],
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
        save_path = os.path.join(
            save_path, "bo", arguments.objective_function, arguments.kernel, arguments.transform)
    else:
        save_path = arguments.save_path
        
    if arguments.objective_function == "npy":
        if arguments.grid_xs_npy_path is None:
            raise ValueError("grid_xs_npy_path must be specified if using npy objective function")
        if arguments.grid_ys_npy_path is None:
            raise ValueError("grid_ys_npy_path must be specified if using npy objective function")

    os.makedirs(save_path, exist_ok=True)

    with jax.experimental.enable_x64(arguments.use_x64):
        if arguments.kernel == "gaussian":
            kernel = kernels.gaussian
        elif arguments.kernel == "matern":
            kernel = kernels.matern
        else:
            raise ValueError(f"Unknown kernel: {arguments.kernel}")
        
        if arguments.objective_function == "npy":
            grid_xs = jnp.load(arguments.grid_xs_npy_path)
            grid_ys = jnp.load(arguments.grid_ys_npy_path)
            jnp.save(os.path.join(save_path, "grid_xs.npy"), grid_xs)
            jnp.save(os.path.join(save_path, "grid_ys.npy"), grid_ys)
            objective_function = objective_functions.MeshGridObjectiveFunction(
                jnp.dstack(jnp.meshgrid(*grid_xs)).reshape(-1, grid_xs.shape[0]), grid_ys.flatten()
            )
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

            objective_function = objective_functions.JitObjectiveFunction(
                objective_function
            )

        if arguments.acquisition_function == "expected_improvement":
            acquisition_function = acquisition_functions.ExpectedImprovement()
        elif arguments.acquisition_function == "lower_confidence_bound":
            acquisition_function = acquisition_functions.LowerConfidenceBound(
                0.5)
        else:
            raise ValueError(
                f"Unknown acquisition function: {arguments.acquisition_function}"
            )

        state = kernels.State(
            jnp.array(math.log(5.0), float),
            jnp.array(math.log(0.5), float),
            jnp.array(math.log(1.0), float),
        )

        bounds = (
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
        bounds = None

        transformer: transformers.Transformer = {
            "standardize": transformers.Standardizer,
            "min_max_scale": transformers.MinMaxScaler,
            "mean_center": transformers.MeanCenterer,
            "standardize_xs_only": transformers.StandardizerXsOnly,
            "mean_center_xs_only": transformers.MeanCentererXsOnly,
            None: transformers.Identity,
        }[arguments.transform]()

        if isinstance(objective_function, objective_functions.MeshGridObjectiveFunction):
            # TODO: Sample some indices! I think I have some code in the old folder for doing this.
            indices = jax.random.randint(jax.random.PRNGKey(0), (arguments.initial_dataset_size, grid_xs.shape[0]), 0, grid_xs.shape[1])
            xs = jnp.take_along_axis(grid_xs.T, indices, axis=0)
            # Honestly I need to learn properly how to index lel.
            ys = grid_ys[*indices.T]
            ticks = tuple(np.array(grid_xs[i]) for i in range(grid_xs.shape[0]))
            grid_xs = jnp.dstack(jnp.meshgrid(*grid_xs)).reshape(-1, grid_xs.shape[0])
            candidates = grid_xs
        else:
            xs = objective_functions.utils.sample(
                jax.random.PRNGKey(0),
                arguments.initial_dataset_size,
                objective_function.dataset_bounds,
            )
            xs_args = tuple(xs[:, i] for i in range(
                xs.shape[1])) if xs.ndim > 1 else (xs,)
            try:
                ys = objective_function.evaluate(jax.random.PRNGKey(0), *xs_args)
            except jaxlib.xla_extension.XlaRuntimeError:
                ys = jnp.concatenate(
                    [
                        objective_function.evaluate(
                            jax.random.PRNGKey(
                                0), *(arg[ii: ii + 1] for arg in xs_args)
                        )
                        for ii in range(arguments.initial_dataset_size)
                    ]
                )

            mesh_grid = objective_functions.utils.get_mesh_grid(
            [
                (boundary, arguments.plot_resolution)
                for boundary in objective_function.dataset_bounds
            ],
            False,
            )
            ticks = tuple(
                np.asarray(
                    objective_functions.utils.get_ticks(
                        boundary, arguments.plot_resolution)
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
                                grid_xs[ii: ii + 1, iii]
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
                    (arguments.plot_resolution,) *
                    len(objective_function.dataset_bounds)
                )

            jnp.save(os.path.join(save_path, "grid_xs.npy"), ticks)
            jnp.save(os.path.join(save_path, "grid_ys.npy"), grid_ys)

            candidates = grid_xs

        dataset = datasets.Dataset(xs, ys)
        (
            transformed_dataset,
            dataset_center,
            dataset_scale,
        ) = transformer.transform_dataset(dataset)

        tried_candidate_indices = []

        optimize = jax.jit(gaussian_process.optimize,
                           static_argnums=(0, 3, 4, 6, 7, 8))
        get_mean_and_std = jax.jit(
            gaussian_process.get_mean_and_std, static_argnums=(0,)
        )
        get_candidate_indices = jax.jit(
            acquisition_function.compute_arg_sort, static_argnums=(0,)
        )
        get_candidate_utilities = jax.jit(
            acquisition_function.__call__, static_argnums=(0,))

        min_y_figure = plt.figure(tight_layout=True, figsize=(4, 4))
        figure = plt.figure(tight_layout=True, figsize=(12, 4))
        try:
            for i in range(arguments.iterations):
                new_state, ok = optimize(
                    kernel,
                    state,
                    transformed_dataset,
                    arguments.optimize_max_iterations,
                    arguments.optimize_tolerance,
                    bounds,
                    arguments.optimize_penalty,
                    use_auto_grad=arguments.use_auto_grad,
                )
                if ok:
                    state = new_state
                else:
                    raise ValueError(f"Optimization failed - {new_state}")

                print(i, state)

                transformed_candidates = transformer.transform_values(
                    candidates,
                    None if dataset_center is None else dataset_center.xs,
                    None if dataset_scale is None else dataset_scale.xs,
                )

                best_candidate_indices = get_candidate_indices(
                    kernel,
                    state,
                    transformed_dataset,
                    transformed_candidates,
                )

                best_candidate_index = None
                for j in range(best_candidate_indices.shape[0]):
                    if best_candidate_indices[j] not in tried_candidate_indices:
                        best_candidate_index = best_candidate_indices[j]
                        tried_candidate_indices.append(best_candidate_index)
                        break
                if best_candidate_index is None:
                    raise ValueError("All candidates have been tried")

                selected_candidate_xs = candidates[best_candidate_index]

                if selected_candidate_xs.ndim == 1:
                    selected_candidate_xs = jnp.expand_dims(
                        selected_candidate_xs, axis=0
                    )

                xs_args = tuple(
                    selected_candidate_xs[:, i]
                    for i in range(selected_candidate_xs.shape[1])
                )

                print(i, selected_candidate_xs)

                key = jax.random.PRNGKey(0)
                selected_candidate_ys = objective_function.evaluate(key, *xs_args)

                dataset = dataset._replace(
                    xs=jnp.concatenate(
                        [dataset.xs, selected_candidate_xs], axis=0),
                    ys=jnp.concatenate(
                        [dataset.ys, selected_candidate_ys], axis=0),
                )
                (
                    transformed_dataset,
                    dataset_center,
                    dataset_scale,
                ) = transformer.transform_dataset(dataset)

                transformed_mean, transformed_std = get_mean_and_std(
                    kernel,
                    state,
                    transformed_dataset,
                    transformer.transform_values(
                        grid_xs,
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

                candidate_utilities = get_candidate_utilities(
                    kernel,
                    state,
                    transformed_dataset,
                    transformed_candidates,
                )

                candidate_utilities = np.asarray(
                    candidate_utilities.reshape(
                        (arguments.plot_resolution,)
                        * len(objective_function.dataset_bounds)
                    )
                )

                figure.clear()
                render.plot(
                    ticks,
                    grid_ys,
                    mean,
                    std,
                    candidate_utilities,
                    np.asarray(dataset.xs),
                    np.asarray(dataset.ys),
                    figure,
                )
                min_y_figure.clear()
                ax = min_y_figure.add_subplot()
                ax.plot([dataset.ys[:i].min() for i in range(1, dataset.ys.shape[0] + 1)])
                step_save_path = os.path.join(save_path, str(i))
                if not os.path.exists(step_save_path):
                    os.makedirs(step_save_path, exist_ok=True)

                figure.savefig(os.path.join(step_save_path, "figure.png"))
                min_y_figure.savefig(os.path.join(step_save_path, "min_y_figure.png"))
                io.write_csv(
                    dataset,
                    os.path.join(step_save_path, "dataset.csv"),
                )
                np.save(os.path.join(step_save_path, "mean.npy"), mean)
                np.save(os.path.join(step_save_path, "std.npy"), std)
                jnp.save(os.path.join(step_save_path, "utility.npy"),
                         candidate_utilities)
        finally:
            plt.close(figure)
