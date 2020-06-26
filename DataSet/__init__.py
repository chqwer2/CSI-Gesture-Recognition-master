from .SignFi import SignFi
# from .transforms import *
import os 

__factory = {
    #'cub': CUB_200_2011,
    #'car': Cars196,
    #'product': Products,
    #'shop': InShopClothes,
    'sign':SignFi,
}


def names():
    return sorted(__factory.keys())


def get_full_name(name):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name].__name__


def create(name, root, set_name, *args, **kwargs):
    if root is not None:
        root = os.path.join(root, get_full_name(name))
    
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root=root , *args, **kwargs)
