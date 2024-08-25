from typing import Union

major: int = 1
minor: int = 4
revision: Union[int, str] = 2

__version__ = ".".join((str(major), str(minor), str(revision)))