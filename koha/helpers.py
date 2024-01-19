import functools
import os


@functools.lru_cache(maxsize=None)
def getenv(key: str, default=0):
    return type(default)(os.getenv(key, default))
