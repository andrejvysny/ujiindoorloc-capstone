"""UJIIndoorLoc capstone — reusable code for building/floor classification."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ujiindoorloc-capstone")
except PackageNotFoundError:  # editable install before metadata generated
    __version__ = "0.0.0"

__all__ = ["__version__"]
