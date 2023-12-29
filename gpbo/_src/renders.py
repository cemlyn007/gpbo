import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib import colors


def plot(
    ticks: tuple[np.ndarray, ...],
    truth: np.ndarray | None,
    mean: np.ndarray,
    std: np.ndarray,
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
            xs,
            ys,
            figure,
        )
    elif len(ticks) == 2:
        # TODO: Implement when we want to plot 2D without truths.
        if truth is None:
            raise NotImplementedError("Truths are required for 2D plots")
        # else...
        plot_2d(
            ticks,
            truth,
            mean,
            std,
            xs,
            ys,
            figure,
        )
    else:
        raise NotImplementedError(f"Cannot plot {len(ticks)} dimensions")


def plot_1d(
    ticks: tuple[np.ndarray, ...],
    truth: np.ndarray | None,
    mean: np.ndarray,
    std: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    figure: matplotlib.figure.Figure,
) -> None:
    (xx,) = ticks

    ax = figure.add_subplot(111)

    ax.plot(xx, mean, c="m")
    ax.plot(xx, mean + std, c="r")
    ax.plot(xx, mean - std, c="b")

    ax.fill_between(xx, mean - std, mean + std, alpha=0.2, color="m")
    ax.scatter(
        xs,
        ys,
        c="g",
        marker="+",
    )

    if truth is not None:
        ax.plot(
            xx,
            truth,
            c="c",
        )
    ax.set_title(f"Gaussian Process Regression")


def plot_2d(
    ticks: tuple[np.ndarray, ...],
    truth: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    figure: matplotlib.figure.Figure,
) -> None:
    (xx, yy) = ticks

    axes = [figure.add_subplot(1, 3, 1 + i) for i in range(3)]

    contour_levels = np.unique(
        np.percentile(np.concatenate([mean.flatten(), truth.flatten()]), np.arange(101))
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
        axes[1].contourf(xx, yy, mean, levels=contour_levels, cmap=cmap, norm=norm)
    except ValueError:
        plot_true = axes[0].contourf(xx, yy, truth, levels=contour_levels, cmap=cmap)
        axes[1].contourf(xx, yy, mean, levels=contour_levels, cmap=cmap)

    try:
        figure.colorbar(plot_true, ax=axes[1])
    except ValueError:
        import traceback

        traceback.print_exc()

        print(np.isinf(xx).any(), np.isinf(yy).any(), np.isinf(mean).any())
        print(np.isnan(xx).any(), np.isnan(yy).any(), np.isnan(mean).any())

    if xs.size:
        axes[0].scatter(xs[:, 0], xs[:, 1], c="b", marker="+")
        axes[1].scatter(xs[:, 0], xs[:, 1], c="b", marker="+")

    axes[0].set_title("True Function")
    axes[1].set_title("Gaussian Process Posterior Mean")

    # Std contour plot
    contour_levels = np.percentile(std, np.arange(101))
    contour_levels = np.unique(contour_levels)
    if len(contour_levels) == 1:
        contour_levels = np.concatenate([np.asarray([0.0, 1.0]), contour_levels])
        contour_levels = np.unique(contour_levels)

    plot_std = axes[2].contourf(xx, yy, std, levels=contour_levels, cmap=cmap)
    if xs.size:
        axes[2].scatter(xs[:, 0], xs[:, 1], c="b", marker="+")
    figure.colorbar(plot_std, ax=axes[2])
    axes[2].set_title("Gaussian Process Posterior Standard Deviation")


def show(
    ticks: tuple[np.ndarray, ...],
    truth: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
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
            xs,
            ys,
            figure,
        )
        plt.show()
    finally:
        plt.close(figure)
