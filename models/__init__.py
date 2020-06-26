from .SimpleNet import Simple_Net
from .ResNet import Res_Net
from .VGG import VGG_

__factory = {
    'Simple-Net':Simple_Net,
    'ResNet':Res_Net,
    'VGG':VGG_,
}

def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown network:", name)
    return __factory[name](*args, **kwargs)
