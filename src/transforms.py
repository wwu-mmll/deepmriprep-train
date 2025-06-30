from fastai.basics import np, torch, store_attr, TensorBase, DisplayedTransform
from fastai.vision.all import RandTransform
from niftiai import TensorImage3d, TensorMask3d


class FlipSagittal(DisplayedTransform):
    order = 10

    def __init__(self, **kwargs):
        store_attr()
        super().__init__(**kwargs)

    def encodes(self, x: (TensorMask3d, TensorImage3d)):
        return x.flip(-3) if x.slices[-3].start + x.slices[-3].stop > 336 else x


class StoreZeroMask(RandTransform):
    order = 45
    def __init__(self, p: float = 1.):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self._zero_mask = TensorBase(b[0] <= 0)

    def encodes(self, x: TensorImage3d):
        x._zero_mask = self._zero_mask
        return x


class ApplyZeroMask(RandTransform):
    order = 82
    def __init__(self, p: float = 1.):
        super().__init__(p=p)
        store_attr()

    def encodes(self, x: TensorImage3d):
        x[x._zero_mask] = 0
        return x


class ScaleIntensity(RandTransform):
    order = 83
    def __init__(self, low: float = .5, high: float = 99.5, p: float = 1.):
        super().__init__(p=p)
        store_attr()

    def encodes(self, x: TensorImage3d):
        x_nonzero = x[~x._zero_mask].cpu()
        x._zero_mask = None
        low = np.percentile(x_nonzero, self.low)
        high = np.percentile(x_nonzero, self.high)
        x = (x - low) / (high - low)
        x[x > 1] = 1 + torch.log10(x[x > 1])
        return x
