# setting backend for matplotlib
try:
    import PySide2  # noqa: F401
except ImportError:
    pass
else:
    import matplotlib

    matplotlib.use("Qt5Agg")
