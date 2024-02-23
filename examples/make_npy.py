if __name__ == "__main__":
    import os
    from gpbo import objective_functions
    import jax.numpy as jnp
    import jax.random
    import jax.experimental
    import numpy as np
    import argparse
    import jax_tqdm
    import platform
    import jaxlib.xla_extension

    argument_parser = argparse.ArgumentParser("Mesh grid data generator")

    argument_parser.add_argument(
        "--resolution",
        type=int,
        default=100,
        help="Mesh grid resolution",
    )
    argument_parser.add_argument(
        "--use_x64", action="store_true", help="Use 64-bit floating point"
    )
    argument_parser.add_argument(
        "--objective_function",
        type=str,
        default="univariate",
        help="Objective function to use, options are univariate and six_hump_camel",
        choices=["univariate", "six_hump_camel", "mnist_1d", "mnist_2d", "mnist_3d"],
    )
    argument_parser.add_argument(
        "--save_path",
        type=str,
        default=None,
    )
    argument_parser.add_argument("--fallback", action="store_true")

    arguments = argument_parser.parse_args()

    if arguments.save_path is None:
        save_path = (
            os.path.join(os.getcwd(), "data.tmp")
            if platform.system() == "Darwin"
            else os.path.join(os.getcwd(), "data")
        )
        save_path = os.path.join(save_path, arguments.objective_function)
    else:
        save_path = arguments.save_path

    os.makedirs(save_path, exist_ok=True)

    with jax.experimental.enable_x64(arguments.use_x64):
        if arguments.objective_function == "univariate":
            objective_function = objective_functions.UnivariateObjectiveFunction()
        elif arguments.objective_function == "six_hump_camel":
            objective_function = objective_functions.SixHumpCamelObjectiveFunction()
        elif arguments.objective_function == "mnist_1d":
            objective_function = objective_functions.MnistObjectiveFunction(
                "/tmp/mnist", False, False, 100, jax.devices()[0]
            )
        elif arguments.objective_function == "mnist_2d":
            objective_function = objective_functions.MnistObjectiveFunction(
                "/tmp/mnist", True, False, 100, jax.devices()[0]
            )
        elif arguments.objective_function == "mnist_3d":
            objective_function = objective_functions.MnistObjectiveFunction(
                "/tmp/mnist", True, True, 100, jax.devices()[0]
            )
        else:
            raise ValueError(
                f"Unknown objective function: {arguments.objective_function}"
            )

        objective_function = objective_functions.JitObjectiveFunction(
            objective_function
        )

        mesh_grid = objective_functions.utils.get_mesh_grid(
            [
                (boundary, arguments.resolution)
                for boundary in objective_function.dataset_bounds
            ],
            False,
        )
        ticks = tuple(
            np.asarray(
                objective_functions.utils.get_ticks(boundary, arguments.resolution)
            )
            for boundary in objective_function.dataset_bounds
        )
        grid_xs = jnp.dstack(mesh_grid).reshape(-1, len(mesh_grid))

        def fallback(grid_xs):
            @jax.jit
            def evaluate(xs: jax.Array) -> jax.Array:

                @jax_tqdm.loop_tqdm(len(grid_xs))
                def body_fun(i, val):
                    return val.at[i].set(
                        objective_function.evaluate(
                            jax.random.PRNGKey(0),
                            *(xs[i, j] for j in range(len(xs[i]))),
                        )
                    )

                return jax.lax.fori_loop(
                    0,
                    len(grid_xs),
                    body_fun,
                    init_val=jnp.zeros((len(grid_xs),)),
                )

            grid_ys = evaluate(grid_xs)
            grid_ys = grid_ys.reshape(
                (arguments.resolution,) * len(objective_function.dataset_bounds)
            )
            return grid_ys

        if arguments.fallback:
            grid_ys = fallback(grid_xs)
        else:
            try:
                grid_ys = objective_function.evaluate(jax.random.PRNGKey(0), *mesh_grid)
            except jaxlib.xla_extension.XlaRuntimeError:
                grid_ys = fallback(grid_xs)

        jnp.save(os.path.join(save_path, "grid_xs.npy"), ticks)
        jnp.save(os.path.join(save_path, "grid_ys.npy"), grid_ys)
