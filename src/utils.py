import torch
import torch.nn.functional as F
ALIGN = True


class LinearElasticity(torch.nn.Module):
    def __init__(self, mu=2., lam=1., refresh_id_grid=False):
        super(LinearElasticity, self).__init__()
        self.mu = mu
        self.lam = lam
        self.id_grid = None
        self.refresh_id_grid = refresh_id_grid

    def forward(self, u):
        if self.id_grid is None or self.refresh_id_grid:
            self.id_grid = create_grid(u.shape[1:4], u.device, u.dtype)
        gradients = jacobi_gradient(u, self.id_grid)
        u_xz, u_xy, u_xx = jacobi_gradient(gradients[None, 2], self.id_grid)
        u_yz, u_yy, u_yx = jacobi_gradient(gradients[None, 1], self.id_grid)
        u_zz, u_zy, u_zx = jacobi_gradient(gradients[None, 0], self.id_grid)
        e_xy = .5 * (u_xy + u_yx)
        e_xz = .5 * (u_xz + u_zx)
        e_yz = .5 * (u_yz + u_zy)
        sigma_xx = 2 * self.mu * u_xx + self.lam * (u_xx + u_yy + u_zz)
        sigma_xy = 2 * self.mu * e_xy
        sigma_xz = 2 * self.mu * e_xz
        sigma_yy = 2 * self.mu * u_yy + self.lam * (u_xx + u_yy + u_zz)
        sigma_yz = 2 * self.mu * e_yz
        sigma_zz = 2 * self.mu * u_zz + self.lam * (u_xx + u_yy + u_zz)
        return (sigma_xx ** 2 + sigma_xy ** 2 + sigma_xz ** 2 +
                sigma_yy ** 2 + sigma_yz ** 2 + sigma_zz ** 2).mean()


def jacobi_gradient(u, id_grid=None):
    if id_grid is None:
        id_grid = create_grid(u.shape[1:4], u.device, u.dtype)
    x = 0.5 * (u + id_grid) * (torch.tensor(u.shape[1:4], device=u.device, dtype=u.dtype) - 1)
    window = torch.tensor([-.5, 0, .5], device=u.device)
    w = torch.zeros((3, 1, 3, 3, 3), device=u.device, dtype=u.dtype)
    w[2, 0, :, 1, 1] = window
    w[1, 0, 1, :, 1] = window
    w[0, 0, 1, 1, :] = window
    x = x.permute(4, 0, 1, 2, 3)
    x = F.conv3d(x, w)
    x = F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate')
    return x.permute(0, 2, 3, 4, 1)


def jacobi_determinant(u, id_grid=None, pad=True):
    gradient = jacobi_gradient(u, id_grid)
    dx, dy, dz = gradient[..., 2], gradient[..., 1], gradient[..., 0]
    jdet0 = dx[2] * (dy[1] * dz[0] - dy[0] * dz[1])
    jdet1 = dx[1] * (dy[2] * dz[0] - dy[0] * dz[2])
    jdet2 = dx[0] * (dy[2] * dz[1] - dy[1] * dz[2])
    jdet = jdet0 - jdet1 + jdet2
    if pad:
        jdet = F.pad(jdet[None, None, 2:-2, 2:-2, 2:-2], (2, 2, 2, 2, 2, 2), mode='replicate')
    return jdet[0, 0]


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


def create_grid(shape, device, dtype):
    return F.affine_grid(torch.eye(4, dtype=dtype, device=device)[None, :3], [1, 3, *shape], align_corners=True)
