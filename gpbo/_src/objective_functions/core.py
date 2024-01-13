import jax
import abc
import typing
import matplotlib.axes


class Boundary[T: int | float](typing.NamedTuple):
    min_value: T
    max_value: T
    dtype: type[T]


class ObjectiveFunction(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, key: jax.Array, *xs: jax.Array) -> jax.Array:
        pass

    @property
    @abc.abstractmethod
    def dataset_bounds(self) -> tuple[Boundary, ...]:
        pass

    @abc.abstractmethod
    def plot(self, axis: matplotlib.axes.Axes, *xs: jax.Array) -> None:
        pass
