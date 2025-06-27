import logging
import typing
from functools import lru_cache as cache

from fibsem.milling.base import MillingStrategy
from fibsem.milling.strategy.standard import StandardMillingStrategy
from fibsem.milling.strategy.overtilt import OvertiltTrenchMillingStrategy


DEFAULT_STRATEGY = StandardMillingStrategy
DEFAULT_STRATEGY_NAME = DEFAULT_STRATEGY.name
BUILTIN_STRATEGIES: typing.Dict[str, typing.Type[MillingStrategy[typing.Any]]] = {
    StandardMillingStrategy.name: StandardMillingStrategy,
    OvertiltTrenchMillingStrategy.name: OvertiltTrenchMillingStrategy,
}
REGISTERED_STRATEGIES: typing.Dict[str, typing.Type[MillingStrategy[typing.Any]]] = {}


def get_strategies() -> typing.Dict[str, typing.Type[MillingStrategy[typing.Any]]]:
    # This order means that builtins > registered > plugins if there are any name clashes
    return {**_get_plugin_strategies(), **REGISTERED_STRATEGIES, **BUILTIN_STRATEGIES}


def get_strategy_names() -> typing.List[str]:
    return list(get_strategies().keys())


def register_strategy(strategy_cls: typing.Type[MillingStrategy[typing.Any]]) -> None:
    global REGISTERED_STRATEGIES
    REGISTERED_STRATEGIES[strategy_cls.name] = strategy_cls


@cache
def _get_plugin_strategies() -> typing.Dict[str, typing.Type[MillingStrategy[typing.Any]]]:
    """
    Import new strategies and append them to the list here

    The plugin logic is based on:
    https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata
    """
    import sys

    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points

    strategies: typing.Dict[str, typing.Type[MillingStrategy[typing.Any]]] = {}
    for strategy_entry_point in entry_points(group="fibsem.strategies"):
        try:
            strategy = strategy_entry_point.load()
            if not issubclass(strategy, MillingStrategy):
                raise TypeError(
                    f"'{strategy_entry_point.value}' is not a subclass of MillingStrategy"
                )
            logging.info("Loaded strategy '%s'", strategy.name)
            strategies[strategy.name] = strategy
        except TypeError as e:
            logging.warning("Invalid strategy found: %s", str(e))
        except Exception:
            logging.error(
                "Unexpected error raised while attempting to import strategy from '%s'",
                strategy_entry_point.value,
                exc_info=True,
            )
    return strategies
