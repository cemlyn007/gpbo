import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib import colors


def plot(
    ticks: tuple[np.ndarray, ...],
    truth: np.ndarray | None,
    mean: np.ndarray,
    std: np.ndarray,
    utility: np.ndarray | None,
    xs: np.ndarray,
    ys: np.ndarray,
    figure: matplotlib.figure.Figure,
) -> None:
    if len(ticks) == 1:
        plot_1d(
            ticks,
            truth,
            mean,
            std,
            utility,
            xs,
            ys,
            figure,
        )
    elif len(ticks) == 2:
        if truth is None:
            plot_2d_without_truth(
                ticks,
                mean,
                std,
                utility,
                xs,
                ys,
                figure,
            )
        else:
            plot_2d(
                ticks,
                truth,
                mean,
                std,
                utility,
                xs,
                ys,
                figure,
            )
    else:
        raise NotImplementedError(f"Cannot plot {len(ticks)} dimensions")

def plot_1d(
    ticks: tuple[np.ndarray],
    truth: np.ndarray | None,
    mean: np.ndarray,
    std: np.ndarray,
    utility: np.ndarray | None,
    xs: np.ndarray,
    ys: np.ndarray,
    figure: matplotlib.figure.Figure,
) -> None:
    (xx,) = ticks

    axes = [figure.add_subplot(1, 2, 1 + i) for i in range(2)]

    axes[0].plot(xx, mean, c="m")
    axes[0].plot(xx, mean + std, c="r")
    axes[0].plot(xx, mean - std, c="b")

    axes[0].fill_between(xx, mean - std, mean + std, alpha=0.2, color="m")
    axes[0].scatter(
        xs,
        ys,
        c="g",
        marker="+",
    )

    if truth is not None:
        axes[0].plot(
            xx,
            truth,
            c="c",
        )
    axes[0].set_title(f"Gaussian Process Regression")

    if utility is not None:
        axes[1].plot(xx, utility, c="m")
        axes[1].set_title(f"Utility Function")



def plot_2d(
    ticks: tuple[np.ndarray, np.ndarray],
    truth: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    utility: np.ndarray | None,
    xs: np.ndarray,
    ys: np.ndarray,
    figure: matplotlib.figure.Figure,
) -> None:
    (xx, yy) = ticks

    if utility is None:
        axes = [figure.add_subplot(1, 3, 1 + i) for i in range(3)]
    else:
        axes = [figure.add_subplot(1, 4, 1 + i) for i in range(4)]

    contour_levels = np.unique(
        np.percentile(np.concatenate(
            [mean.flatten(), truth.flatten()]), np.arange(101))
    )
    if len(contour_levels) == 1:
        contour_levels = np.asarray([0.0] + contour_levels.tolist())
    elif len(contour_levels) == 0:
        contour_levels = np.array([0.0, 1.0])
    norm = colors.SymLogNorm(
        linthresh=0.03,
        linscale=0.03,
        vmin=contour_levels.min().item(),
        vmax=contour_levels.max().item(),
        base=10,
    )
    cmap = "hot"
    try:
        plot_true = axes[0].contourf(
            xx, yy, truth, levels=contour_levels, cmap=cmap, norm=norm
        )
        axes[1].contourf(xx, yy, mean, levels=contour_levels,
                         cmap=cmap, norm=norm)
    except ValueError:
        plot_true = axes[0].contourf(
            xx, yy, truth, levels=contour_levels, cmap=cmap)
        axes[1].contourf(xx, yy, mean, levels=contour_levels, cmap=cmap)

    figure.colorbar(plot_true, ax=axes[1])

    if xs.size:
        axes[0].scatter(xs[:, 0], xs[:, 1], c="b", marker="+")
        axes[1].scatter(xs[:, 0], xs[:, 1], c="b", marker="+")

    axes[0].set_title("True Function")
    axes[1].set_title("Gaussian Process Posterior Mean")

    # Std contour plot
    contour_levels = np.percentile(std, np.arange(101))
    contour_levels = np.unique(contour_levels)
    if len(contour_levels) == 1:
        contour_levels = np.concatenate(
            [np.asarray([0.0, 1.0]), contour_levels])
        contour_levels = np.unique(contour_levels)

    plot_std = axes[2].contourf(xx, yy, std, levels=contour_levels, cmap=cmap)
    if xs.size:
        axes[2].scatter(xs[:, 0], xs[:, 1], c="b", marker="+")
    figure.colorbar(plot_std, ax=axes[2])
    axes[2].set_title("Gaussian Process Posterior Standard Deviation")

    if utility is not None:
        axes[3].contourf(xx, yy, utility)
        axes[3].set_title("Utility Function")


def plot_2d_without_truth(
    ticks: tuple[np.ndarray, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    utility: np.ndarray | None,
    xs: np.ndarray,
    ys: np.ndarray,
    figure: matplotlib.figure.Figure,
) -> None:
    (xx, yy) = ticks

    if utility is None:
        axes = [figure.add_subplot(1, 2, 1 + i) for i in range(2)]
    else:
        axes = [figure.add_subplot(1, 3, 1 + i) for i in range(3)]

    contour_levels = np.unique(np.percentile(mean.flatten(), np.arange(101)))
    if len(contour_levels) == 1:
        contour_levels = np.asarray([0.0] + contour_levels.tolist())
    elif len(contour_levels) == 0:
        contour_levels = np.array([0.0, 1.0])
    norm = colors.SymLogNorm(
        linthresh=0.03,
        linscale=0.03,
        vmin=contour_levels.min().item(),
        vmax=contour_levels.max().item(),
        base=10,
    )
    cmap = "hot"
    try:
        mean_plot = axes[0].contourf(
            xx, yy, mean, levels=contour_levels, cmap=cmap, norm=norm
        )
    except ValueError:
        mean_plot = axes[0].contourf(
            xx, yy, mean, levels=contour_levels, cmap=cmap)

    figure.colorbar(mean_plot, ax=axes[0])

    if xs.size:
        axes[0].scatter(xs[:, 0], xs[:, 1], c="b", marker="+")
        axes[1].scatter(xs[:, 0], xs[:, 1], c="b", marker="+")

    axes[0].set_title("Gaussian Process Posterior Mean")

    # Std contour plot
    contour_levels = np.percentile(std, np.arange(101))
    contour_levels = np.unique(contour_levels)
    if len(contour_levels) == 1:
        contour_levels = np.concatenate(
            [np.asarray([0.0, 1.0]), contour_levels])
        contour_levels = np.unique(contour_levels)

    plot_std = axes[1].contourf(xx, yy, std, levels=contour_levels, cmap=cmap)
    if xs.size:
        axes[0].scatter(xs[:, 0], xs[:, 1], c="b", marker="+")
    figure.colorbar(plot_std, ax=axes[1])
    axes[1].set_title("Gaussian Process Posterior Standard Deviation")
    if utility is not None:
        axes[2].contourf(xx, yy, utility)
        axes[2].set_title("Utility Function")

def show(
    ticks: tuple[np.ndarray, ...],
    truth: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    utility: np.ndarray | None,
    xs: np.ndarray,
    ys: np.ndarray,
) -> None:
    figure = plt.figure()
    try:
        plot(
            ticks,
            truth,
            mean,
            std,
            utility,
            xs,
            ys,
            figure,
        )
        plt.show()
    finally:
        plt.close(figure)
