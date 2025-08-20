from .HardNegativeNLLLoss import HardNegativeNLLLoss
from .InfoNCE import InfoNCE

def load_loss(loss_class, scale):
    if loss_class == "HardNegativeNLLLoss":
        loss_cls = HardNegativeNLLLoss(scale)
    elif loss_class == "InfoNCE":
        loss_cls = InfoNCE(scale)
    else:
        raise ValueError(f"Unknown loss class {loss_class}")
    return loss_cls
