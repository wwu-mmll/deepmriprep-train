from fastai.basics import torch, random, store_attr, F
from fastai.vision.all import RandTransform
from mriaug.utils import to_ndim
from niftiai import TensorImage3d


class ScaledChiNoise3d(RandTransform):  # to compensate for initial affine transformation
    order = 50
    def __init__(self, max_intensity: float = .1, max_downsample: float = 3., max_dof: int = 3, p: float = .5, batch: bool = False):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        x = b[0]
        dof = random.randint(1, self.max_dof)
        shape = list(x.shape[:-3]) + [min(int(s / (1 + self.max_downsample * random.random())), s) for s in x.shape[-3:]]
        self._noise = to_ndim(self.max_intensity, x.ndim) * torch.randn([*(shape[(1 if self.batch else 0):]), dof], device=x.device)
        # Only works with single channel images!
        self._noise = F.interpolate(self._noise[:, 0].permute(0, 4, 1, 2, 3), x.shape[-3:], mode='trilinear').permute(0, 2, 3, 4, 1)[:, None]

    def encodes(self, x: TensorImage3d):
        return ((x[..., None] + self._noise) ** 2).mean(-1).sqrt()


class Blur3d(RandTransform):
    order = 52

    def __init__(self, max_sigma=.5, kernel_size=7, p=.1):
        super().__init__(p=p)
        store_attr()

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        kernel = gaussian_smoothing_kernel(self.kernel_size, random.random() * self.max_sigma)
        self._kernel = kernel[None, None].repeat(b[0].shape[1], 1, 1, 1, 1).to(b[0].device)

    def encodes(self, x: TensorImage3d):
        return F.conv3d(F.pad(x, [self._kernel.shape[-1] // 2] * 6, mode='reflect'), self._kernel, groups=x.shape[1])


def gaussian_smoothing_kernel(kernel_size, sigma, normalize=True):
    kernel_size = 3 * [kernel_size] if isinstance(kernel_size, int) else kernel_size
    sigma = 3 * [sigma] if isinstance(sigma, float) else sigma
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * (2 * torch.pi)**.5) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)
    return kernel / kernel.sum() if normalize else kernel


def gauss_smoothing(x, sigma=3., kernel_size=3):
    kernel = gaussian_smoothing_kernel(kernel_size, sigma).to(x.device)
    kernel = kernel[None, None].repeat(x.shape[1], 1, 1, 1, 1)
    x = F.pad(x, [kernel_size // 2] * 6, mode='reflect')
    return F.conv3d(x.type(torch.float32), kernel, groups=x.shape[1]).type(x.dtype)
