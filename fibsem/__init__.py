
try:
    import importlib.metadata
    __version__ = importlib.metadata.version('fibsem')
except ModuleNotFoundError:
    __version__ = "unknown"

    