import ants
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from niftiai import TensorImage3d
from spline_resize import resize, grid_sample
ALIGN = True
data_path = 'data'


def min_max(x, low=.005, high=.995):
    mask = x > 0
    low, high = np.percentile(x[mask].cpu(), 100 * low), np.percentile(x[mask].cpu(), 100 * high)
    x = (x - low) / (high - low)
    x[x > 1] = 1 + torch.log10(x[x > 1])
    return x.clamp(min=0)


def one_hot(o, n_classes=4):
    o = o.clip(max=n_classes - 1)
    one_h = torch.zeros((n_classes, *o.shape[1:]), dtype=torch.float32, device=o.device)
    for c in range(n_classes):
        mask = o[0].gt(c - 1) & o[0].le(c + 1)
        one_h[c, mask] = 1 - (o[0][mask] - c).abs()
    return one_h


if __name__ == '__main__':
    data_subdirs = ['bet', 'bc', 'img_05mm', 'img_05mm_minmax', 'img_05mm_minmax_raw', 'img_075mm_minmax',
                    'img_075mm_minmax_raw', 'p0_05mm', 'nogm', 'p0_075mm', 'p']
    for subdir in data_subdirs: Path(f'{data_path}/{subdir}').mkdir(exist_ok=True)
    shape_05mm = (339, 411, 339)
    shape_075mm = (224, 256, 224)
    shape_15mm = (113, 137, 113)
    nib_affine_05mm = np.array([[.5, 0, 0, -84], [0, .5, 0, -120], [0, 0, .5, -72], [0, 0, 0, 0]])
    nib_affine_075mm = np.array([[.75, 0, 0, -84], [0, .75, 0, -120], [0, 0, .75, -72], [0, 0, 0, 0]])
    nib_affine_15mm = np.array([[1.5, 0, 0, -84], [0, 1.5, 0, -120], [0, 0, 1.5, -72], [0, 0, 0, 0]])
    fns = pd.read_csv('data/csvs/openneuro_hd.csv').filename
    for fn in tqdm(fns):
        affine = pd.read_csv(f'{data_path}/affine/{fn}.csv')
        affine = torch.linalg.inv(torch.from_numpy(affine.values).cuda()).float()
        im = TensorImage3d.create(f'{data_path}/t1/{fn}.nii.gz').cuda()
        header = im.header
        p0 = TensorImage3d.create(f'{data_path}/t1/CAT12.8.2/mri/p0{fn}.nii').cuda()
        im_bet = im.clone()
        mask = p0 <= 0
        im_bet[mask] = 0  # brain extraction
        im_bet.affine, im_bet.header = im.affine, header
        bet_fp = f'{data_path}/bet/{fn}.nii.gz'
        im_bet.save(bet_fp)
        bc_fp = f'{data_path}/bc/{fn}.nii.gz'
        ants.n4_bias_field_correction(ants.image_read(bet_fp)).to_file(bc_fp)  # bias correction
        im_bc = TensorImage3d.create(bc_fp).cuda()
        grid = F.affine_grid(affine[None, :3], [1, 3, *shape_05mm], align_corners=ALIGN)
        header.set_data_dtype(np.float32)
        mask = F.grid_sample(mask[None].float(), grid, mode='nearest')[0, :, 1:-2, 15:-12, :336]
        zeromask = mask > 0
        im = grid_sample(im[None], grid, align_corners=ALIGN)[0, :, 1:-2, 15:-12, :336]
        im_bc = grid_sample(im_bc[None], grid, align_corners=ALIGN, mask_value=0)[0, :, 1:-2, 15:-12, :336]
        im_bet = grid_sample(im_bet[None], grid, align_corners=ALIGN, mask_value=0)[0, :, 1:-2, 15:-12, :336]
        im = TensorImage3d(im, affine=nib_affine_05mm, header=header)
        im.save(f'{data_path}/img_05mm/{fn}.nii.gz')  # run CAT12 on these files
        im_bc = TensorImage3d(min_max(im_bc), affine=nib_affine_05mm, header=header)
        im_bc.save(f'{data_path}/img_05mm_minmax/{fn}.nii.gz')  # train input(=brain extracted+bias corrected @ 0.5mm)
        im_bet = TensorImage3d(min_max(im_bet), affine=nib_affine_05mm, header=header)
        im_bet.save(f'{data_path}/img_05mm_minmax_raw/{fn}.nii.gz')  # eval input(=brain extracted @ 0.5mm)
        im_bc = resize(im_bc[None], shape_075mm, align_corners=ALIGN, mask_value=0)[0]  # interpolate to 0.75mm
        im_bet = resize(im_bet[None], shape_075mm, align_corners=ALIGN, mask_value=0)[0]  # interpolate to 0.75mm
        TensorImage3d(im_bc, affine=nib_affine_075mm, header=header).save(f'{data_path}/img_075mm_minmax/{fn}.nii.gz')
        TensorImage3d(im_bet, affine=nib_affine_075mm, header=header).save(f'{data_path}/img_075mm_minmax_raw/{fn}.nii.gz')
    print('Run CAT12 on all "img_05mm/..."-files, then run commented code block at the end of this script')
    # for fn in tqdm(fns):
    #     affine = pd.read_csv(f'{data_path}/affine/{fn}.csv')
    #     affine = torch.linalg.inv(torch.from_numpy(affine.values).cuda()).float()
    #     im = TensorImage3d.create(f'{data_path}/t1/{fn}.nii.gz').cuda()
    #     header = im.header
    #     p0 = TensorImage3d.create(f'{data_path}/t1/CAT12.8.2/mri/p0{fn}.nii').cuda()
    #     mask = p0 <= 0
    #     grid = F.affine_grid(affine[None, :3], [1, 3, *shape_05mm], align_corners=ALIGN)
    #     header.set_data_dtype(np.float32)
    #     mask = F.grid_sample(mask[None].float(), grid, mode='nearest')[0, :, 1:-2, 15:-12, :336]
    #     zeromask = mask > 0
    #     p0 = TensorImage3d.create(f'{data_path}/img_05mm/CAT12.8.2/mri/p0{fn}.nii').cuda()
    #     p0_header = p0.header
    #     p0[zeromask] = 0
    #     p0.save(f'{data_path}/p0_05mm/{fn}.nii.gz')  # train target for patchwise brain segm.
    #     p1 = TensorImage3d.create(f'{data_path}/img_05mm/CAT12.8.2/mri/p1{fn}.nii').cuda()
    #     p1[zeromask] = 0
    #     p1_pre = one_hot(p0[None])[2]
    #     nogm = ((p1_pre - p1) > .015)
    #     TensorImage3d(nogm, affine=nib_affine_05mm, header=header).save(f'{data_path}/nogm/{fn}.nii.gz')  # train target for nogm
    #     p0 = resize(p0[None], shape_075mm, align_corners=ALIGN, mask_value=0)[0]
    #     p0 = TensorImage3d(p0, affine=nib_affine_075mm, header=p0_header)
    #     p0.save(f'{data_path}/p0_075mm/{fn}.nii.gz')  # train target for brain segm.
    #     p1 = TensorImage3d.create(f'{data_path}/t1/CAT12.8.2/mri/p1{fn}.nii').cuda()
    #     p2 = TensorImage3d.create(f'{data_path}/t1/CAT12.8.2/mri/p2{fn}.nii').cuda()
    #     p3 = TensorImage3d.create(f'{data_path}/t1/CAT12.8.2/mri/p3{fn}.nii').cuda()
    #     header = p1.header
    #     header.set_data_dtype(np.float32)
    #     grid = F.affine_grid(affine[None, :3], (1, 3, *shape_15mm), align_corners=True)
    #     p = grid_sample(torch.cat([p1, p2, p3])[None], grid, align_corners=True, mask_value=0)[0].clip(0, 1)
    #     p.affine, p.header = nib_affine_15mm, header
    #     p.save(f'{data_path}/p/{fn}.nii.gz')  # train input for syn registration
