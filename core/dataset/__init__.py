from .Si_predict import *
from .UCI import *

def get_dataset(name: str):
    return globals()[f"{name}"]()


__all__ = ['get_dataset']

