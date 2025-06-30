from fastai.basics import pd, torch, set_seed, F, Path, Learner
from fastai.data.all import DataBlock, ColReader, ColSplitter, DataLoaders
from niftiai.data import ImageBlock3d
from src.models import SyMNet
from src.utils import ALIGN, create_grid, jacobi_determinant, LinearElasticity
path = '.'  # if data_path is absolute(=starts with "/") set path = '/'
data_path = 'data'
Path(f'{data_path}/models').mkdir(exist_ok=True)


class SyNDiffeo:
    def __init__(self, time_steps=7):
        self.time_steps = time_steps
        self.grid = None

    def apply_flows(self, v_xy, v_yx):
        half_flows = self.diffeomorphic_transform(torch.cat([v_xy, v_yx, -v_xy, -v_yx]))
        full_flows = self.composition_transform(half_flows[:2], half_flows[2:].flip(0))
        return {'xy_half': half_flows[:1], 'yx_half': half_flows[1:2], 'xy_full': full_flows[:1], 'yx_full': full_flows[1:2]}

    def diffeomorphic_transform(self, v):
        v = v / (2 ** self.time_steps)
        for i in range(self.time_steps):
            v = v + self.spatial_transform(v, v)
        return v

    def composition_transform(self, v1, v2):
        return v2 + self.spatial_transform(v1, v2)

    def spatial_transform(self, x, v):
        if self.grid is None:
            self.grid = create_grid(v.shape[2:], x.device, dtype=x.dtype)
        return F.grid_sample(x, self.grid + v.permute(0, 2, 3, 4, 1), align_corners=True, padding_mode='border')


class SupervisedLoss:
    def __init__(self, beta=.1, lambda_syn=2e-5):
        self.beta = beta
        self.lambda_syn = lambda_syn
        self.elast = LinearElasticity(mu=2., lam=1.)
        self.syn = SyNDiffeo(time_steps=7)

    def __call__(self, f_pred, f_xy_targ, f_yx_targ):
        v_xy_pred, v_yx_pred, x, y = f_pred
        flows_pred = self.syn.apply_flows(v_xy_pred, v_yx_pred)
        flows_targ = self.syn.apply_flows(f_xy_targ, f_yx_targ)
        flow_xy = flows_pred['xy_full'].permute(0, 2, 3, 4, 1)
        flow_yx = flows_pred['yx_full'].permute(0, 2, 3, 4, 1)
        half_flow_xy = flows_pred['xy_half'].permute(0, 2, 3, 4, 1)
        half_flow_yx = flows_pred['yx_half'].permute(0, 2, 3, 4, 1)
        elast_xy = self.elast(flow_xy)
        elast_yx = self.elast(flow_yx)
        wj_pred = jacobi_determinant(flow_xy)
        wj_targ = jacobi_determinant(flows_targ['xy_full'].permute(0, 2, 3, 4, 1))
        xy = F.grid_sample(x, flow_xy + self.syn.grid, align_corners=ALIGN, mode='bilinear')
        yx = F.grid_sample(y, flow_yx + self.syn.grid, align_corners=ALIGN, mode='bilinear')
        half_xy = F.grid_sample(x, half_flow_xy + self.syn.grid, align_corners=ALIGN, mode='bilinear')
        half_yx = F.grid_sample(y, half_flow_yx + self.syn.grid, align_corners=ALIGN, mode='bilinear')
        mse_syn = ((half_xy - half_yx) ** 2).mean() + ((xy - y) ** 2).mean() + ((yx - x) ** 2).mean()
        loss_syn = mse_syn + self.lambda_syn * (elast_xy + elast_yx)
        loss_v = ((v_xy_pred - f_xy_targ) ** 2).mean() + ((v_yx_pred - f_yx_targ) ** 2).mean()
        loss_j = ((wj_pred - wj_targ) ** 2).mean()
        return loss_v + loss_j + self.beta * loss_syn


def get_dls(df, img_col, mask_col, v_xy_col, v_yx_col, valid_col):
    dblock = DataBlock(blocks=[ImageBlock3d(), ImageBlock3d(), ImageBlock3d(), ImageBlock3d()],
                       get_x=(ColReader(img_col), ColReader(mask_col)),
                       get_y=(ColReader(v_xy_col), ColReader(v_yx_col)),
                       splitter=ColSplitter(valid_col),
                       n_inp=2)
    return DataLoaders.from_dblock(dblock, df, path='/', bs=1)


if __name__ == '__main__':
    set_seed(1)
    df = pd.read_csv(f'{data_path}/csvs/openneuro_hd.csv')[:5]
    df['img'] = f'{data_path}/p/' + df.filename + '.nii.gz[:,:,:,:2]'
    df['mask'] = f'{data_path}/templates/Template_4_GS.nii[:,:,:,:2]'
    df['v_xy'] = f'{data_path}/v_xy/' + df.filename + '.nii.gz'
    df['v_yx'] = f'{data_path}/v_yx/' + df.filename + '.nii.gz'
    loss_func = SupervisedLoss()

    # train on full openneuro-hd dataset
    df_total = df.copy()
    df_total.loc[len(df_total)] = df_total.loc[0]
    df_total['is_valid'] = (len(df_total) - 1) * [0] + [1]
    dls = get_dls(df_total, img_col='img', mask_col='mask', v_xy_col='v_xy', v_yx_col='v_yx', valid_col='is_valid')
    learn = Learner(dls=dls, model=SyMNet(), loss_func=SupervisedLoss())
    learn.fit_one_cycle(50, 1e-3)
    torch.save(learn.model.state_dict(), f'{data_path}/models/warp_model.pth')
    # train cross validation
    for fold in range(5):
        set_seed(1)
        df['is_valid'] = df.fold == fold
        dls = get_dls(df, img_col='img', mask_col='mask', v_xy_col='v_xy', v_yx_col='v_yx', valid_col='is_valid')
        learn = Learner(dls=dls, model=SyMNet(), loss_func=SupervisedLoss())
        learn.fit_one_cycle(50, 1e-3)
        torch.save(learn.model.state_dict(), f'{data_path}/models/warp_model_fold{fold}.pth')
