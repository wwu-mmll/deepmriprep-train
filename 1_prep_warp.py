import torch
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from niftiai import TensorImage3d
from src.utils import ALIGN, create_grid, gauss_smoothing, LinearElasticity
TEMPLATE_SIZE = (168, 204, 168)
TEMPLATE_SHAPE = (113, 137, 113)
TEMPLATE_ORIGIN = (72, 120, 84)
data_path = 'data'


def compose_affine(translation, rotation, zoom, shear):
    affine = torch.eye(4, device=translation.device)
    ZS = torch.diag(zoom)
    ZS[0, -2:] = shear[:2]
    ZS[1, 2] = shear[2]
    affine[:3, :3] = torch.mm(rotation, ZS)
    affine[:3, 3] = translation
    return affine


def fill_nans(iy_disp, grid, nans):
    iy_disp_org = iy_disp.clone()
    nans_in_template = F.grid_sample(nans[None, None].float(), grid, align_corners=ALIGN)[0, 0]
    nans_in_template = nans_in_template > .0001
    empty_mask = (grid[0, ..., 0].abs() > 1) + (grid[0, ..., 1].abs() > 1) + (grid[0, ..., 2].abs() > 1) > 0
    nan_mask = nans_in_template | empty_mask
    elast = LinearElasticity()
    fill = torch.nn.Parameter(torch.zeros(int(nan_mask.sum()), 3, device=iy_disp.device, requires_grad=ALIGN))
    opt = torch.optim.Adam([fill], lr=1e-3)
    for i in range(100):
        opt.zero_grad()
        iy_disp[nan_mask] = fill
        loss = elast(iy_disp[None])
        loss.backward(retain_graph=True)
        opt.step()
    iy_disp = iy_disp.detach()
    iy_disp = gauss_smoothing(iy_disp.permute(3, 0, 1, 2)[None])[0].permute(1, 2, 3, 0)
    iy_disp_org[nan_mask] = iy_disp[nan_mask]
    return iy_disp_org


def decompose_iy(iy, mask, nans):
    iy = (iy + torch.tensor(TEMPLATE_ORIGIN, device=iy.device)) / torch.tensor(TEMPLATE_SIZE, device=iy.device)
    iy = (iy * 2) - 1
    translation = torch.nn.Parameter(torch.zeros(3).cuda(), requires_grad=True)
    rotation = torch.nn.Parameter(torch.eye(3).cuda(), requires_grad=True)
    zoom = torch.nn.Parameter(torch.ones(3).cuda(), requires_grad=True)
    shear = torch.nn.Parameter(torch.zeros(3).cuda(), requires_grad=True)
    opt = torch.optim.Adam([translation, rotation, zoom, shear], lr=3e-2)
    pbar = tqdm(range(100), disable=True)
    for _ in pbar:
        opt.zero_grad()
        affine = compose_affine(translation, rotation, zoom, shear)
        affine_grid = F.affine_grid(affine[None, :3], [1, 3, *iy.shape[:3]], align_corners=ALIGN)
        loss = ((iy[None, mask, :] - affine_grid[:, mask, :]) ** 2).mean()
        pbar.set_description(f'{loss.item()}')
        loss.backward()
        opt.step()
    iy_disp = iy.detach() - affine_grid.detach()[0]
    inv_affine = torch.linalg.inv(affine.detach())
    inv_affine_grid = F.affine_grid(inv_affine[None, :3], [1, 3, *TEMPLATE_SHAPE], align_corners=ALIGN)
    iy_disp = iy_disp.permute(3, 0, 1, 2)[None]
    iy_disp_in_template_space = F.grid_sample(iy_disp, inv_affine_grid, align_corners=ALIGN)[0].permute(1, 2, 3, 0)
    iy_disp_in_template_space = fill_nans(iy_disp_in_template_space, inv_affine_grid, nans)
    return iy_disp_in_template_space.detach(), affine


def mse(x, y):
    return ((x - y) ** 2).mean()


class SyN:
    def __init__(self, time_steps=7, factor_diffeo=.1, sim_func=mse, mu=2., lam=1., optimizer=torch.optim.Adam):
        self.time_steps = time_steps
        self.factor_diffeo = factor_diffeo
        self.sim_func = sim_func
        self.reg_func = LinearElasticity(mu, lam, refresh_id_grid=True)
        self.optimizer = optimizer
        self.grid = None

    def fit_xy(self, targ_f_yx, iterations, learning_rate):
        x = 0 * targ_f_yx[:, :1]
        y = 0 * targ_f_yx[:, :1]
        self.grid = create_grid(x.shape[2:], x.device, dtype=x.dtype)
        v_xy = torch.zeros((x.shape[0], 3, *x.shape[2:]), device=x.device, dtype=x.dtype)
        v_yx = torch.zeros((x.shape[0], 3, *x.shape[2:]), device=x.device, dtype=x.dtype)
        v_xy = torch.nn.Parameter(v_xy, requires_grad=True)
        v_yx = torch.nn.Parameter(v_yx, requires_grad=True)
        optimizer = self.optimizer([v_xy, v_yx], learning_rate)
        for i in range(iterations):
            optimizer.zero_grad()
            images, flows = self.apply_flows(x, y, v_xy, v_yx)
            loss = self.sim_func(targ_f_yx, flows['yx_full'])
            loss.backward()
            optimizer.step()
        return flows['xy_full'].detach(), v_xy, v_yx, loss.detach().item()

    def apply_flows(self, x, y, v_xy, v_yx):
        half_flows = self.diffeomorphic_transform(torch.cat([v_xy, v_yx, -v_xy, -v_yx]))
        half_images = self.spatial_transform(torch.cat([x, y]), half_flows[:2])
        full_flows = self.composition_transform(half_flows[:2], half_flows[2:].flip(0))
        full_images = self.spatial_transform(torch.cat([x, y]), full_flows)
        images = {'xy_half': half_images[:1], 'yx_half': half_images[1:2],
                  'xy_full': full_images[:1], 'yx_full': full_images[1:2]}
        flows = {'xy_half': half_flows[:1], 'yx_half': half_flows[1:2],
                 'xy_full': full_flows[:1], 'yx_full': full_flows[1:2]}
        return images, flows

    def diffeomorphic_transform(self, flow):
        flow = self.factor_diffeo * flow / (2 ** self.time_steps)
        for i in range(self.time_steps):
            flow = flow + self.spatial_transform(flow, flow)
        return flow

    def composition_transform(self, flow_1, flow_2):
        return flow_2 + self.spatial_transform(flow_1, flow_2)

    def spatial_transform(self, x, flow):
        return F.grid_sample(x.type(torch.float32), self.grid.type(torch.float32) + flow.permute(0, 2, 3, 4, 1),
                             align_corners=ALIGN, padding_mode='border')


def preprocess_cat12_registration(p0_filepaths, iy_filepaths, y_filepaths, dest_dir=None):
    nib_affine = np.array([[-1.5,0,0,84], [0,1.5,0,-120], [0,0,1.5,-72], [0,0,0,0]])
    for p0_fpath, iy_fpath, y_fpath in tqdm(zip(p0_filepaths, iy_filepaths, y_filepaths), total=len(y_filepaths)):
        p0 = TensorImage3d.create(p0_fpath)[0].cuda()
        iy = nib.load(iy_fpath)
        iy = TensorImage3d.create(iy.get_fdata(), affine=iy.affine, header=iy.header).cuda()
        y = nib.load(y_fpath)
        y = TensorImage3d.create(y.get_fdata(), affine=y.affine, header=y.header).cuda()
        y, iy = y[..., 0, :].flip(3), iy[..., 0, :].flip(3)
        brainmask = p0 > .001
        nans = torch.isnan(iy[..., 0])
        iy[nans, :] = 0
        mask = (~brainmask & ~nans)
        iy_disp, affine = decompose_iy(iy, mask, nans)
        iy_disp = iy_disp.permute(3, 0, 1, 2)
        syn = SyN()
        y_disp, v_xy, v_yx, lss = syn.fit_xy(iy_disp[None], iterations=100, learning_rate=1e-1)
        if dest_dir is not None:
            filename = p0_fpath.split('/')[-1].split('.')[0][2:]
            pd.DataFrame(affine).to_csv(f'{dest_dir}/affine/{filename}.csv', index=False)
            TensorImage3d(iy_disp, affine=nib_affine, header=iy.header).save(f'{dest_dir}/flow_yx/{filename}.nii.gz')
            TensorImage3d(y_disp[0], affine=nib_affine, header=iy.header).save(f'{dest_dir}/flow_xy/{filename}.nii.gz')
            TensorImage3d(v_yx[0], affine=nib_affine, header=iy.header).save(f'{dest_dir}/v_yx/{filename}.nii.gz')
            TensorImage3d(v_xy[0], affine=nib_affine, header=iy.header).save(f'{dest_dir}/v_xy/{filename}.nii.gz')


if __name__ == '__main__':
    df = pd.read_csv(f'{data_path}/csvs/openneuro_hd.csv')
    cat_dir = f'{data_path}/t1/CAT12.8.2'
    p0_fps = cat_dir + '/mri/p0' + df.filename + '.nii'
    iy_fps = cat_dir + '/mri/iy_' + df.filename + '.nii'
    y_fps = cat_dir + '/mri/y_' + df.filename + '.nii'
    for subdir in ['affine', 'flow_xy', 'flow_yx', 'v_xy', 'v_yx']: Path(f'{data_path}/{subdir}').mkdir(exist_ok=True)
    preprocess_cat12_registration(p0_fps, iy_fps, y_fps, dest_dir=data_path)
